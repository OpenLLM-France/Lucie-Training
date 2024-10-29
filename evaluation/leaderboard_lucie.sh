#!/bin/bash

## Lucie
export LUCIE_TOKENIZER_PATH=/lustre/fsn1/projects/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_65k_grouped/tokenizer
export CHECKPOINT_PATH=$ALL_CCFRSCRATCH/trained_models/Lucie/pretrained/transformers_checkpoints
export OUTPUT_PATH=out/leaderboard/lucie
export OUT_PREFIX=__lustre__fsn1__projects__rech__qgz__commun__trained_models__Lucie__pretrained__transformers_checkpoints__global_step

export i=25000
while [ $i -le 700000 ]; do
	if [ ! -d ${OUTPUT_PATH}/${OUT_PREFIX}${i} ]; then
		srun --exclusive --ntasks=1 lm-eval \
			--model_args "pretrained=${CHECKPOINT_PATH}/global_step${i},tokenizer=${LUCIE_TOKENIZER_PATH},dtype=bfloat16,add_bos_token=True" \
			--tasks leaderboard \
			--batch_size auto \
			--output_path $OUTPUT_PATH \
			--seed 42 &
	fi
	i=$((i + 25000))
done

wait