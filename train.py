import argparse
import datetime
import logging
import numpy as np
import os
import sys
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from models.dual_encoder import DualEncoder
from torch.nn.init import xavier_uniform_
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers.optimization import get_linear_schedule_with_warmup
from utils import parsing
from utils.data_utils import DualEncoderDataset
from utils.train_utils import get_lr, grad_norm, log_rank_0, param_count, \
    param_norm, set_seed, setup_logger


def get_train_parser():
    parser = argparse.ArgumentParser("train")
    parsing.add_common_args(parser)
    parsing.add_train_args(parser)

    return parser


def init_dist(args):
    if args.local_rank != -1:
        dist.init_process_group(backend=args.backend,
                                init_method='env://',
                                timeout=datetime.timedelta(0, 7200))
        torch.cuda.set_device(args.local_rank)
        torch.backends.cudnn.benchmark = False

    if dist.is_initialized():
        logging.info(f"Device rank: {dist.get_rank()}")
        sys.stdout.flush()


def get_model(args, model_class, device):
    state = {}
    if args.load_from:
        log_rank_0(f"Loading pretrained state from {args.load_from}")
        state = torch.load(args.load_from, map_location=torch.device("cpu"))
        pretrain_args = state["args"]
        pretrain_args.local_rank = args.local_rank
        parsing.log_args(pretrain_args, phase="pretraining")

        model = model_class(pretrain_args, device)
        pretrain_state_dict = state["state_dict"]
        pretrain_state_dict = {k.replace("module.", ""): v for k, v in pretrain_state_dict.items()}
        model.load_state_dict(pretrain_state_dict)
        log_rank_0("Loaded pretrained model state_dict.")
    else:
        model = model_class(args, device)
        # for p in model.parameters():
        #     if p.dim() > 1 and p.requires_grad:
        #         xavier_uniform_(p)

    model.to(device)
    if args.local_rank != -1:
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True
        )
        log_rank_0("DDP setup finished")

    return model, state


def get_optimizer_and_scheduler(args, model, state, warmup_step: int, total_step: int):
    if args.optimizer == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer {args.optimizer}")

    if args.scheduler == "linear_with_warmup":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_step,
            num_training_steps=total_step
        )
    else:
        raise ValueError(f"Unsupported scheduler {args.scheduler}")

    if state and args.resume:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        log_rank_0("Loaded pretrained optimizer and scheduler state_dicts.")

    return optimizer, scheduler


def init_loader(args, dataset, batch_size: int,
                shuffle: bool = False, epoch: int = None):
    if shuffle:
        dataset.shuffle()
    dataset.batch(batch_size=batch_size)

    if args.local_rank != -1:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        if epoch is not None:
            sampler.set_epoch(epoch)
    else:
        sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)

    # implementation of collation is offloaded to .batch()
    loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=lambda _batch: _batch[0],
        # pin_memory=True
    )

    return loader


def _optimize(args, model, optimizer, scheduler):
    nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
    optimizer.step()
    scheduler.step()
    g_norm = grad_norm(model)
    model.zero_grad(set_to_none=True)

    return g_norm


def train_main(args):
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = args.device

    init_dist(args)
    parsing.log_args(args, phase="training")

    os.makedirs(args.save_dir, exist_ok=True)
    if args.model == "base":
        model_class = DualEncoder
        dataset_class = DualEncoderDataset
    else:
        raise ValueError(f"Model {args.model} not supported!")

    model, state = get_model(args, model_class, device)
    log_rank_0(model)
    log_rank_0(f"Number of parameters = {param_count(model)}")

    train_dataset = dataset_class(args, file=args.train_file)
    val_dataset = dataset_class(args, file=args.train_file)

    total_step = train_dataset.size // args.train_batch_size // \
        int(os.environ["NUM_GPUS_PER_NODE"]) * args.epoch
    warmup_step = int(total_step * args.warmup_ratio)
    log_rank_0(f"Total step: {total_step}, warmup_step: {warmup_step}")

    optimizer, scheduler = get_optimizer_and_scheduler(
        args, model, state, warmup_step=warmup_step, total_step=total_step)

    step = state["step"] if state else 0
    accum = 0
    g_norm = 0
    losses, accs = [], []
    o_start = time.time()
    log_rank_0("Start training")

    for epoch in range(args.epoch):
        log_rank_0(f"Epoch: {epoch + 1} / {args.epoch}")
        model.train()
        model.zero_grad(set_to_none=True)
        train_loader = init_loader(args, train_dataset,
                                   batch_size=args.train_batch_size,
                                   shuffle=True,
                                   epoch=epoch)
        for batch_idx, train_batch in enumerate(train_loader):
            loss, acc = model(train_batch, mode="train")
            loss.backward()
            losses.append(loss.item())
            accs.append(acc.item() * 100)

            accum += 1
            if accum == args.accumulation_count:
                _optimize(args, model, optimizer, scheduler)
                accum = 0
                step += 1

            if (accum == 0) and (step > 0) and (step % args.log_iter == 0):
                log_rank_0(f"Step {step}, loss: {np.mean(losses)}, "
                           f"acc: {np.mean(accs): .4f}, p_norm: {param_norm(model): .4f}, "
                           f"g_norm: {g_norm: .4f}, lr: {get_lr(optimizer): .6f}, "
                           f"elapsed time: {time.time() - o_start: .0f}")
                losses, accs = [], []

            if (accum == 0) and (step > 0) and (step % args.eval_iter == 0):
                model.eval()
                val_count = 100
                val_losses, val_accs = [], []

                val_loader = init_loader(args, val_dataset,
                                         batch_size=args.val_batch_size,
                                         shuffle=True,
                                         epoch=None)
                with torch.no_grad():
                    for val_idx, val_batch in enumerate(val_loader):
                        if val_idx >= val_count:
                            break
                        val_loss, val_acc = model(val_batch, mode="val")
                        val_losses.append(val_loss.item())
                        val_accs.append(val_acc.item() * 100)

                log_rank_0(f"Validation (in batch) at step {step}, "
                           f"val loss: {np.mean(val_losses)}, "
                           f"val acc: {np.mean(val_accs): .4f}")
                model.train()

            # Important: saving only at one node or the ckpt would be corrupted!
            if dist.is_initialized() and dist.get_rank() > 0:
                continue

            if (step > 0) and (step % args.save_iter == 0):
                n_iter = step // args.save_iter - 1
                log_rank_0(f"Saving at step {step}")
                state = {
                    "args": args,
                    "step": step,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()
                }
                torch.save(state, os.path.join(args.save_dir, f"model.{step}_{n_iter}.pt"))

        # lastly
        if (args.accumulation_count > 1) and (accum > 0):
            _optimize(args, model, optimizer, scheduler)
            accum = 0
            step += 1

        if args.local_rank != -1:
            dist.barrier()


if __name__ == "__main__":
    train_parser = get_train_parser()
    args = train_parser.parse_args()
    args.local_rank = int(os.environ["LOCAL_RANK"])

    # set random seed
    set_seed(args.seed)

    # logger setup
    logger = setup_logger(args)

    # maximize display for debugging
    np.set_printoptions(threshold=sys.maxsize)
    torch.set_printoptions(profile="full")

    train_main(args)
