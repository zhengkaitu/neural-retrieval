import json
import os
import pandas as pd
from tqdm import tqdm


def preprocess():
    data_path = "./data/USPTO_condition_MIT"
    preprocessed_path = "preprocessed/USPTO_condition_MIT_smiles"
    os.makedirs(preprocessed_path, exist_ok=True)

    corpus_file = os.path.join(data_path, "USPTO_rxn_corpus.csv")
    print(f"Loading corpus from {corpus_file}")
    corpus = pd.read_csv(corpus_file)
    corpus.set_index("id", inplace=True)

    for phase in ["train", "val", "test"]:
        print(f"Matching SMILES with texts in corpus for: {phase}")
        rxn_file = os.path.join(data_path, f"USPTO_condition_{phase}.csv")
        rxn_df = pd.read_csv(rxn_file)

        ofn = os.path.join(preprocessed_path, f"{phase}.jsonl")
        with open(ofn, "w") as of:
            for i, row in tqdm(rxn_df.iterrows()):
                query_id = row["id"]
                query = row["canonical_rxn"]

                instance = {
                    "query_id": query_id,
                    "query": query,
                    "positive_passages": [
                        {
                            "docid": query_id,
                            "title": corpus.at[query_id, "heading_text"],
                            "text": corpus.at[query_id, "paragraph_text"]
                        }
                    ],
                    "negative_passages": []
                }
                of.write(f"{json.dumps(instance, ensure_ascii=False)}\n")


if __name__ == "__main__":
    preprocess()
