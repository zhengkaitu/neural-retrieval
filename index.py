import hnswlib
import numpy as np
import os
import sys
import time
from train import get_train_parser
from utils import parsing
from utils.train_utils import log_rank_0, set_seed, setup_logger


def index_main(args):
    parsing.log_args(args, phase="index")
    os.makedirs("./indexed/USPTO_condition_MIT_smiles", exist_ok=True)
    o_start = time.time()

    encoding_file = "./encoded/USPTO_condition_MIT_smiles/p_encodings.npy"
    output_file = "./indexed/USPTO_condition_MIT_smiles/corpus.bin"
    # encoding_file = "./encoded/USPTO_condition_MIT_smiles/test_q_encodings.npy"

    p_encodings = np.load(encoding_file)
    p_ids = np.arange(len(p_encodings))
    log_rank_0(f"Loaded encodings from {encoding_file}, "
               f"no. of element: {len(p_encodings)}, start indexing with hnsw")

    # Declaring index
    p = hnswlib.Index(space='ip', dim=512)  # possible options are l2, cosine or ip

    # Initializing index - the maximum number of elements should be known beforehand
    p.init_index(max_elements=3000000, ef_construction=2500, M=35, random_seed=args.seed)

    # Element insertion (can be called several times):
    p.add_items(p_encodings, p_ids, num_threads=64)

    log_rank_0(f"Done indexing, time: {time.time() - o_start}, saving to {output_file}")
    p.save_index(output_file)


if __name__ == "__main__":
    train_parser = get_train_parser()
    args = train_parser.parse_args()
    args.local_rank = -1

    set_seed(args.seed)

    # logger setup
    logger = setup_logger(args)

    # maximize display for debugging
    np.set_printoptions(threshold=sys.maxsize)

    index_main(args)
