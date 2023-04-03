from utils.train_utils import log_rank_0


def log_args(args, phase: str):
    log_rank_0(f"Logging {phase} arguments")
    for k, v in vars(args).items():
        log_rank_0(f"**** {k} = *{v}*")


def add_common_args(parser):
    group = parser.add_argument_group("Meta")
    group.add_argument("--model", help="Model architecture",
                       choices=["base"], type=str, default="")
    group.add_argument("--data_name", help="Data name", type=str, default="")
    group.add_argument("--seed", help="Random seed", type=int, default=42)
    group.add_argument("--max_src_len", help="Max source length", type=int, default=1024)
    group.add_argument("--max_tgt_len", help="Max target length", type=int, default=1024)
    group.add_argument("--num_workers", help="No. of workers", type=int, default=1)
    group.add_argument("--verbose", help="Whether to enable verbose debugging", action="store_true")

    group = parser.add_argument_group("Paths")
    group.add_argument("--log_file", help="Preprocess log file", type=str, default="")
    group.add_argument("--preprocess_output_path", help="Path for saving preprocessed outputs",
                       type=str, default="")
    group.add_argument("--save_dir", help="Path for saving checkpoints", type=str, default="")


def add_preprocess_args(parser):
    """Placeholder"""
    group = parser.add_argument_group("Preprocessing options")


def add_train_args(parser):
    group = parser.add_argument_group("Training options")
    # file paths
    group.add_argument("--train_file", help="Train file", type=str, default="")
    group.add_argument("--val_file", help="Val file", type=str, default="")
    group.add_argument("--load_from", help="Checkpoint to load", type=str, default="")
    # model params
    group.add_argument("--q_encoder_type", help="Query encoder type", type=str, default="")
    group.add_argument("--q_encoder_name", help="Query encoder name", type=str, default="")
    group.add_argument("--q_pool_type", help="Query pooling type", type=str, default="sum")
    group.add_argument("--p_encoder_type", help="Passage encoder type", type=str, default="")
    group.add_argument("--p_encoder_name", help="Passage encoder name", type=str, default="")
    group.add_argument("--p_pool_type", help="Passage pooling type", type=str, default="sum")
    group.add_argument("--output_size", help="Vector output size", type=int, default=512)
    # training params
    group.add_argument('--local_rank', type=int, default=-1, metavar='N', help='Local process rank')
    group.add_argument("--resume", help="Whether to resume training", action="store_true")
    group.add_argument("--backend", help="Backend for DDP", type=str, choices=["gloo", "nccl"], default="nccl")
    group.add_argument("--epoch", help="Number of training epochs", type=int, default=20)
    group.add_argument("--optimizer", help="optimizer type", type=str, default="AdamW")
    group.add_argument("--lr", help="Learning rate", type=float, default=0.0)
    group.add_argument("--beta1", help="Adam beta 1", type=float, default=0.9)
    group.add_argument("--beta2", help="Adam beta 2", type=float, default=0.998)
    group.add_argument("--eps", help="Adam epsilon", type=float, default=1e-9)
    group.add_argument("--weight_decay", help="Adam weight decay", type=float, default=1e-2)
    group.add_argument("--scheduler", help="scheduler type", type=str, default="linear_with_warmup")
    group.add_argument("--warmup_ratio", help="Warmup ratio", type=float, default=0.02)
    group.add_argument("--clip_norm", help="Max norm for gradient clipping", type=float, default=20.0)
    group.add_argument("--dropout", help="Hidden dropout", type=float, default=0.1)
    group.add_argument("--train_batch_size", help="Batch size for train", type=int, default=4096)
    group.add_argument("--val_batch_size", help="Batch size for validation", type=int, default=4096)
    group.add_argument("--accumulation_count", help="No. of batches for gradient accumulation", type=int, default=1)
    group.add_argument("--log_iter", help="No. of steps per logging", type=int, default=100)
    group.add_argument("--eval_iter", help="No. of steps per evaluation", type=int, default=100)
    group.add_argument("--save_iter", help="No. of steps per saving", type=int, default=100)
