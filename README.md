# neural-retrieval

# Environment setup
```
bash -i scripts/setup.sh
conda activate nn-retrieval
```

# Preprocess
```
python preprocess_smiles.py
```

# Train
```
sh scripts/train_1_baseline.sh
```
Note that there is a hardcode in the scripts that kills ALL lingering processes with nccl. Otherwise, these zombie processes won't exit.

# Encode
```
sh scripts/encode_1_baseline.sh
```

# Index
```
sh scripts/index_1_baseline.sh
```

# Retrieve (and score)
```
python retrieve.py --log_file="retrieve.1_baseline.log"
```
