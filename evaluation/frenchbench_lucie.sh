#!/bin/bash

## Lucie
export LUCIE_TOKENIZER_PATH=/lustre/fsn1/projects/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_65k_grouped/tokenizer
export CHECKPOINT_PATH=$ALL_CCFRSCRATCH/trained_models/Lucie/pretrained/transformers_checkpoints

export i=25000
while [ $i -le 600000 ]; do
	srun --exclusive --ntasks=1 lm-eval \
		--model_args "pretrained=${CHECKPOINT_PATH}/global_step${i},tokenizer=${LUCIE_TOKENIZER_PATH},dtype=bfloat16,add_bos_token=True" \
		--tasks french_bench_mc \
        --num_fewshot 5 \
		--batch_size auto \
		--output_path out/french_bench/lucie \
		--seed 42 &
	i=$((i + 25000))
done

wait