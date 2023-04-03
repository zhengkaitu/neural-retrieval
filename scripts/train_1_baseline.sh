#!/bin/bash
pkill -f nccl

export NUM_GPUS_PER_NODE=4
export NUM_NODES=1
export NODE_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=5678
export TOKENIZERS_PARALLELISM=false

export MODEL=base
export LR=2e-5
export BATCH_SIZE=32
export ACCUM_COUNT=1
export DROPOUT=0.0
export N_WORKERS=20
export LOG_ITER=20
export EVAL_ITER=5000
export SAVE_ITER=10000


CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
  --nproc_per_node="$NUM_GPUS_PER_NODE" \
  --nnodes="$NUM_NODES" \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  train.py \
  --backend=nccl \
  --model="$MODEL" \
  --data_name="USPTO_condition_MIT" \
  --load_from="" \
  --train_file="./preprocessed/USPTO_condition_MIT_smiles/train.jsonl" \
  --val_file="./preprocessed/USPTO_condition_MIT_smiles/val.jsonl" \
  --log_file="train.1_baseline.log" \
  --save_dir="./checkpoints/1_baseline" \
  --q_encoder_type="hugging_face" \
  --q_encoder_name="seyonec/ChemBERTa-zinc-base-v1" \
  --q_pool_type="mean" \
  --p_encoder_type="hugging_face" \
  --p_encoder_name="allenai/scibert_scivocab_uncased" \
  --p_pool_type="mean" \
  --output_size=512 \
  --dropout="$DROPOUT" \
  --seed=42 \
  --epoch=20 \
  --warmup_ratio=0.02 \
  --optimizer="AdamW" \
  --lr="$LR" \
  --weight_decay=0.0 \
  --scheduler="linear_with_warmup" \
  --clip_norm=20.0 \
  --train_batch_size="$BATCH_SIZE" \
  --val_batch_size="$BATCH_SIZE" \
  --accumulation_count="$ACCUM_COUNT" \
  --num_workers="$N_WORKERS" \
  --log_iter="$LOG_ITER" \
  --eval_iter="$EVAL_ITER" \
  --save_iter="$SAVE_ITER"
