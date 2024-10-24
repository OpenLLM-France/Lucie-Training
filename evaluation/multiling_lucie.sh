#!/bin/bash

## Lucie
export LUCIE_TOKENIZER_PATH=/lustre/fsn1/projects/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_65k_grouped/tokenizer
export CHECKPOINT_PATH=$ALL_CCFRSCRATCH/trained_models/Lucie/pretrained/transformers_checkpoints

export i=25000
while [ $i -le 600000 ]; do
	srun --exclusive --ntasks=1 lm-eval \
		--model_args "pretrained=${CHECKPOINT_PATH}/global_step${i},tokenizer=${LUCIE_TOKENIZER_PATH},dtype=bfloat16,add_bos_token=True" \
		--tasks m_mmlu_fr,m_mmlu_es,m_mmlu_de,m_mmlu_it,arc_fr,arc_es,arc_de,arc_it \
        --num_fewshot 25 \
		--batch_size auto \
		--output_path out/okapi/lucie \
		--seed 42 &
	i=$((i + 25000))
done

wait