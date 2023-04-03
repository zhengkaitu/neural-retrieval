import argparse
import hnswlib
import json
import numpy as np
import os
import pandas as pd
import sys
import time
import logging
from datetime import datetime
from multiprocessing import Pool
from tqdm import tqdm
from utils import parsing

index_file = "./indexed/USPTO_condition_MIT_smiles/corpus.bin"
p = hnswlib.Index(space='ip', dim=512)
p.load_index(index_file)
p.set_ef(1000)
p.set_num_threads(64)

corpus_file = "./data/USPTO_condition_MIT/USPTO_rxn_corpus.csv"
df = pd.read_csv(corpus_file)
doc_ids = df["id"].tolist()

log_rank_0 = logging.info


def setup_logger(args, warning_off: bool = False):
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


def calculate_recalls(line: str) -> np.ndarray:
    recall = np.zeros(100, dtype=np.float32)
    d = json.loads(line.strip())
    gt = d["positive_passages"][0]["docid"]
    for i, neg in enumerate(d["negative_passages"]):
        if neg["docid"] == gt:
            recall[i:] = 1.0
            break

    return recall


def retrieve_helper(_args) -> str:
    line, encoding = _args
    data = np.expand_dims(encoding, 0)

    labels, distances = p.knn_query(data, k=100)

    label = labels[0]
    distance = distances[0]

    d = json.loads(line.strip())

    for l, score in zip(label, distance):
        d["negative_passages"].append({
            "id": l.item(),
            "docid": doc_ids[l],
            "score": score.item()
        })
    output_line = json.dumps(d)

    return output_line


def retrieve_main(args):
    parsing.log_args(args, phase="retrieve")
    os.makedirs("./retrieved/USPTO_condition_MIT_smiles", exist_ok=True)
    o_start = time.time()

    if args.do_retrieve:
        # pool = Pool(64)

        for phase in ["train", "val", "test"]:
            log_rank_0(f"Retrieving for phase: {phase}")

            preprocessed_file = f"./preprocessed/USPTO_condition_MIT_smiles/{phase}.jsonl"
            encoding_file = f"./encoded/USPTO_condition_MIT_smiles/{phase}_q_encodings.npy"
            retrieved_file = f"./retrieved/USPTO_condition_MIT_smiles/{phase}.jsonl"

            q_encodings = np.load(encoding_file)
            with open(preprocessed_file, "r") as f:
                lines = f.readlines()
            assert len(lines) == len(q_encodings)
            log_rank_0(f"Loaded encodings from {encoding_file}, and lines from {preprocessed_file}")

            with open(retrieved_file, "w") as of:
                for line, encoding in tqdm(zip(lines, q_encodings)):
                    output_line = retrieve_helper((line, encoding))
                # for output_line in tqdm(pool.imap(retrieve_helper, zip(lines, q_encodings))):
                    of.write(f"{output_line.strip()}\n")

            log_rank_0(f"Retrieval done, time: {time.time() - o_start}")

        # pool.close()
        # pool.join()

    if args.do_score:
        pool = Pool(64)

        for phase in ["train", "val", "test"]:
            log_rank_0(f"Calculating recalls for phase: {phase}")

            retrieved_file = f"./retrieved/USPTO_condition_MIT_smiles/{phase}.jsonl"
            with open(retrieved_file, "r") as f:
                lines = f.readlines()

            recalls = tqdm(pool.imap(calculate_recalls, lines))
            recalls = np.stack(list(recalls), axis=0)

            mean_recalls = np.mean(recalls, axis=0)
            for n in range(100):
                log_rank_0(f"Top {n + 1} recall: {mean_recalls[n]}")
            log_rank_0("\n")

        pool.close()
        pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("retrieve")
    parser.add_argument("--log_file", type=str)
    parser.add_argument("--data_name", default="", type=str)
    args = parser.parse_args()

    args.local_rank = -1
    args.do_retrieve = True
    args.do_score = True

    # set_seed(args.seed)

    # logger setup
    logger = setup_logger(args)

    # maximize display for debugging
    np.set_printoptions(threshold=sys.maxsize)

    retrieve_main(args)
