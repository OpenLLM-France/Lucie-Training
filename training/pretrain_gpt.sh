#!/bin/bash

# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPT_ARGS="
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 4 \
    --global-batch-size 8 \
    --lr 0.00015 \
    --train-iters 500000 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16
"

DATA_ARGS="
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path OpenLLM-France/Lucie-tokenizer-v2.4-space_prefix_all \
    --data-impl mmap \
    --data-path /gpfsscratch/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_2.4-space_prefix_all/Wikipedia--fr--026_text_document
"

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"

DISTRIBUTED_ARGS="
   --DDP-impl torch
"
MASTER_PORT=$(python find_free_port.py)
echo "The following free master port was found: $MASTER_PORT"
# /gpfsscratch/rech/qgz/urc37ho/lucie-cache
torchrun --master_port $MASTER_PORT $1/pretrain_gpt.py \
    --data-cache-path $2 \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS 