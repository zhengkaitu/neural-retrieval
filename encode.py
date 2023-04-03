import json
import numpy as np
import os
import pandas as pd
import sys
import time
import torch
from models.dual_encoder import DualEncoder
from tqdm import tqdm
from train import get_model, get_train_parser
from utils import parsing
from utils.data_utils import DualEncoderBatch
from utils.train_utils import log_rank_0, param_count, set_seed, setup_logger


def encode_main(args):
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = args.device

    parsing.log_args(args, phase="encode")

    os.makedirs("./encoded/USPTO_condition_MIT_smiles", exist_ok=True)
    if args.model == "base":
        model_class = DualEncoder
    else:
        raise ValueError(f"Model {args.model} not supported!")

    model, state = get_model(args, model_class, device)
    log_rank_0(model)
    log_rank_0(f"Number of parameters = {param_count(model)}")
    batch_size = args.train_batch_size

    o_start = time.time()
    model.eval()

    for phase in ["train", "val", "test"]:
        log_rank_0(f"Start encoding for phase {phase}")
        q_encodings = []

        file = os.path.join(f"./preprocessed/USPTO_condition_MIT_smiles", f"{phase}.jsonl")
        with open(file, "r") as f:
            qs = [json.loads(line.strip())["query"] for line in f]
            for i in tqdm(range(0, len(qs), batch_size)):
                batch = DualEncoderBatch(qs=qs[i:i+batch_size])
                q_encodings.append(model.encode_q(batch).detach().cpu().numpy())

        q_encodings = np.concatenate(q_encodings, axis=0)
        output_file = f"./encoded/USPTO_condition_MIT_smiles/{phase}_q_encodings.npy"
        log_rank_0(f"Saving to {output_file}, shape: {q_encodings.shape} time: {time.time() - o_start}")
        np.save(output_file, q_encodings)

    file = "./data/USPTO_condition_MIT/USPTO_rxn_corpus.csv"
    df = pd.read_csv(file)
    ps = df["paragraph_text"].tolist()

    log_rank_0(f"Start encoding for corpus")
    p_encodings = []

    for i in tqdm(range(0, len(ps), batch_size)):
        batch = DualEncoderBatch(ps=ps[i:i+batch_size])
        p_encodings.append(model.encode_p(batch).detach().cpu().numpy())

    p_encodings = np.concatenate(p_encodings, axis=0)
    output_file = "./encoded/USPTO_condition_MIT_smiles/p_encodings.npy"
    log_rank_0(f"Saving to {output_file}, shape: {p_encodings.shape} time: {time.time() - o_start}")
    np.save(output_file, p_encodings)


if __name__ == "__main__":
    train_parser = get_train_parser()
    args = train_parser.parse_args()
    args.local_rank = -1                    # turn off DDP, 不知道有什么幺蛾子

    # set random seed
    set_seed(args.seed)

    # logger setup
    logger = setup_logger(args)

    # maximize display for debugging
    np.set_printoptions(threshold=sys.maxsize)
    torch.set_printoptions(profile="full")

    encode_main(args)
