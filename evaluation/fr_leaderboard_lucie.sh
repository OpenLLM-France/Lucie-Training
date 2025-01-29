#!/bin/bash

module purge
module load arch/h100
module load anaconda-py3/2024.06
conda activate /lustre/fsn1/projects/rech/qgz/uzq54wg/envs/french_eval
# pip install -e .[vllm,ifeval,math]

## Lucie
LUCIE_TOKENIZER_PATH=/lustre/fsn1/projects/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_65k_grouped/tokenizer
CHECKPOINT_PATH=$ALL_CCFRSCRATCH/trained_models/Lucie/pretrained/transformers_checkpoints
OUTPUT_PATH=out/fr_leaderboard/lucie
OUT_PREFIX=__lustre__fsn1__projects__rech__qgz__commun__trained_models__Lucie__pretrained__transformers_checkpoints__global_step

CHECKPOINT_PATH=$ALL_CCFRSCRATCH/trained_models/Lucie/pretrained/transformers_checkpoints
LUCIE_TOKENIZER_PATH=/lustre/fsn1/projects/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_65k_grouped/tokenizer

export TRANSFORMERS_OFFLINE=0
export HF_DATASETS_OFFLINE=0

lm-eval \
	--model vllm \
	--model_args "pretrained=${CHECKPOINT_PATH}/global_step753851,tokenizer=${LUCIE_TOKENIZER_PATH},tensor_parallel_size=1,dtype=bfloat16,gpu_memory_utilization=0.8" \
	--tasks=leaderboard-fr \
	--batch_size=auto \
	--output_path $OUTPUT_PATH \
	--log_samples 
