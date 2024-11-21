#!/bin/bash

module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0
conda activate lucie_eval

## Lucie
LUCIE_TOKENIZER_PATH=/lustre/fsn1/projects/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_65k_grouped/tokenizer
CHECKPOINT_PATH=$ALL_CCFRSCRATCH/trained_models/Lucie/pretrained/transformers_checkpoints
OUTPUT_PATH=out/leaderboard/lucie
OUT_PREFIX=__lustre__fsn1__projects__rech__qgz__commun__trained_models__Lucie__pretrained__transformers_checkpoints__global_step

i=25000
while [ $i -le 750000 ]; do
	if [ ! -d ${OUTPUT_PATH}/${OUT_PREFIX}${i} ]; then
		echo "processing step $i..."
		srun --exclusive --ntasks=1 lm-eval \
			--model_args "pretrained=${CHECKPOINT_PATH}/global_step${i},tokenizer=${LUCIE_TOKENIZER_PATH},dtype=bfloat16,add_bos_token=True" \
			--tasks leaderboard \
			--batch_size auto \
			--output_path $OUTPUT_PATH \
			--seed 42 &
	fi
	i=$((i + 25000))
done

# Final
if [ ! -d ${OUTPUT_PATH}/${OUT_PREFIX}753851 ]; then
	srun --exclusive --ntasks=1 lm-eval \
		--model_args "pretrained=${CHECKPOINT_PATH}/global_step753851,tokenizer=${LUCIE_TOKENIZER_PATH},dtype=bfloat16,add_bos_token=True" \
		--tasks leaderboard \
		--batch_size auto \
		--output_path $OUTPUT_PATH \
		--seed 42 &
fi

# Extension
CP_PATH=$ALL_CCFRSCRATCH/trained_models/Lucie/extension_rope20M/transformers_checkpoints/global_step1220

if [ ! -d ${OUTPUT_PATH}/__lustre__fsn1__projects__rech__qgz__commun__trained_models__Lucie__extension_rope20M__transformers_checkpoints__global_step1220 ]; then
	echo "processing extension..."
	echo $CP_PATH
	srun --exclusive --ntasks=1 lm-eval \
		--model_args "pretrained=${CP_PATH},tokenizer=${LUCIE_TOKENIZER_PATH},dtype=bfloat16,add_bos_token=True" \
		--tasks leaderboard \
		--batch_size auto \
		--output_path $OUTPUT_PATH \
		--seed 42 &
fi
wait