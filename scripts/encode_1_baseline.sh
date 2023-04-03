#!/bin/bash
pkill -f encode.py

export TOKENIZERS_PARALLELISM=false
export MODEL=base
export BATCH_SIZE=32


CUDA_VISIBLE_DEVICES=0 python encode.py \
  --model="$MODEL" \
  --data_name="USPTO_condition_MIT" \
  --load_from="checkpoints/1_baseline/model.80000_7.pt" \
  --log_file="encode.1_baseline.log" \
  --q_encoder_type="hugging_face" \
  --q_encoder_name="seyonec/ChemBERTa-zinc-base-v1" \
  --q_pool_type="mean" \
  --p_encoder_type="hugging_face" \
  --p_encoder_name="allenai/scibert_scivocab_uncased" \
  --p_pool_type="mean" \
  --output_size=512 \
  --seed=42 \
  --train_batch_size="$BATCH_SIZE"
