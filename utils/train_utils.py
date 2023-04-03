import logging
import math
import numpy as np
import os
import random
import sys
import torch
import torch.nn as nn
from datetime import datetime
from rdkit import RDLogger


def param_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def param_norm(m):
    return math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))


def grad_norm(m):
    return math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_logger(args, warning_off: bool = False):
    if warning_off:
        RDLogger.DisableLog("rdApp.*")
    else:
        RDLogger.DisableLog("rdApp.warning")

    os.makedirs(f"./logs/{args.data_name}", exist_ok=True)
    dt = datetime.strftime(datetime.now(), "%y%m%d-%H%Mh")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(f"./logs/{args.data_name}/{args.log_file}.{dt}")
    sh = logging.StreamHandler(sys.stdout)
    fh.setLevel(logging.INFO)
    sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger


def log_rank_0(message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            logging.info(message)
            sys.stdout.flush()
    else:
        logging.info(message)
        sys.stdout.flush()


def log_tensor(tensor, tensor_name: str, shape_only=False):
    log_rank_0(f"--------------------------{tensor_name}--------------------------")
    if not shape_only:
        log_rank_0(tensor)
    if isinstance(tensor, torch.Tensor):
        log_rank_0(tensor.shape)
    elif isinstance(tensor, np.ndarray):
        log_rank_0(tensor.shape)
    elif isinstance(tensor, list):
        try:
            for item in tensor:
                log_rank_0(item.shape)
        except Exception as e:
            log_rank_0(f"Error: {e}")
            log_rank_0("List items are not tensors, skip shape logging.")
