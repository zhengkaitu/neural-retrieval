import json
import random
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import List
from utils.train_utils import log_rank_0


class DualEncoderBatch:
    def __init__(self, qs: List[str] = None, ps: List[str] = None):
        # assert len(qs) == len(ps), f"qs: {qs} \n ps: {ps}"
        self.qs = qs
        self.ps = ps
        self.size = len(qs) if qs is not None else len(ps)

    def to(self, device):
        raise NotImplementedError

    def __repr__(self):
        return f"qs: {self.qs} \n" \
               f"ps: {self.ps} \n"


class DualEncoderDataset(Dataset):
    def __init__(self, args, file: str):
        self.qps = []
        self.batches = []

        with open(file, "r") as f:
            for i, line in enumerate(tqdm(f)):
                # if i >= 13000:
                #     break
                instance = json.loads(line.strip())
                q = instance["query"]
                p = instance["positive_passages"][0]["text"]

                self.qps.append((q, p))

        self.size = i

        log_rank_0(f"Loaded and initialized DualEncoderDataset, size: {self.size}")

    def shuffle(self):
        random.shuffle(self.qps)

    def batch(self, batch_size: int):
        self.batches = []
        # dropping the last partial batch for robustness
        for i in range(0, self.size - batch_size, batch_size):
            qs = []
            ps = []
            for q, p in self.qps[i:i+batch_size]:
                qs.append(q)
                ps.append(p)
            self.batches.append(DualEncoderBatch(qs=qs, ps=ps))

    def __getitem__(self, index: int) -> DualEncoderBatch:
        return self.batches[index]

    def __len__(self):
        return len(self.batches)
