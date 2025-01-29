#!/bin/bash

# Paths
BASE_CHECKPOINT_PATH=$ALL_CCFRSCRATCH/trained_models/Lucie
BASE_OUTPUT_PATH=out/french_bench/lucie
BASE_LUCIE_TOKENIZER_PATH=/lustre/fsn1/projects/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_65k_grouped/tokenizer

# Function to run `lm-eval` for a given model checkpoint
run_evaluation() {
    local cp_path=$1              # Checkpoint path
    local tokenizer_path=$2       # Tokenizer path (default: base path)
    local output_subdir=$3        # Output directory
    local peft_path=$4            # PEFT path (optional)
    local additional_args=${5:-""} # Additional arguments (optional)

    # If PEFT is provided, include it in the model arguments
    local peft_args=""
    if [ -n "$peft_path" ]; then
        peft_args=",peft=${peft_path}"
    fi

    if [ ! -d "${BASE_OUTPUT_PATH}/${output_subdir}" ]; then
        echo "Processing checkpoint: $cp_path"
        srun --exclusive --ntasks=1 lm-eval \
            --model_args "pretrained=${cp_path},tokenizer=${tokenizer_path},dtype=bfloat16${peft_args},add_bos_token=True" \
			${additional_args} \
            --tasks french_bench_mc \
            --num_fewshot 5 \
            --batch_size auto \
            --output_path $BASE_OUTPUT_PATH \
            --seed 42 &
    fi
}

# Evaluate Lucie models at intervals
OUTPUT_PREFIX=__lustre__fsn1__projects__rech__qgz__commun__trained_models__Lucie__pretrained__transformers_checkpoints__global_step
CHECKPOINT_INTERVAL_START=25000
CHECKPOINT_INTERVAL_END=750000
CHECKPOINT_INTERVAL_STEP=25000

for i in $(seq $CHECKPOINT_INTERVAL_START $CHECKPOINT_INTERVAL_STEP $CHECKPOINT_INTERVAL_END); do
    run_evaluation "${BASE_CHECKPOINT_PATH}/pretrained/transformers_checkpoints/global_step${i}" \
        $BASE_LUCIE_TOKENIZER_PATH "${OUTPUT_PREFIX}${i}"
done

# Evaluate the final checkpoint
run_evaluation "${BASE_CHECKPOINT_PATH}/pretrained/transformers_checkpoints/global_step753851" \
    $BASE_LUCIE_TOKENIZER_PATH "${OUTPUT_PREFIX}753851"

# Evaluate extension checkpoint
EXTENSION_CHECKPOINT=${BASE_CHECKPOINT_PATH}/extension_rope20M/transformers_checkpoints/global_step1220
EXTENSION_OUTPUT=__lustre__fsn1__projects__rech__qgz__commun__trained_models__Lucie__extension_rope20M__transformers_checkpoints__global_step1220
run_evaluation $EXTENSION_CHECKPOINT $BASE_LUCIE_TOKENIZER_PATH $EXTENSION_OUTPUT

# Evaluate annealing checkpoint
i=1
while [ $i -le 4 ]; do
	ANNEALING_CHECKPOINT=${BASE_CHECKPOINT_PATH}/annealing/mix_$i/transformers_checkpoints/global_step9
	ANNEALING_OUTPUT=__lustre__fsn1__projects__rech__qgz__commun__trained_models__Lucie__annealing__mix_${i}__transformers_checkpoints__global_step9
	run_evaluation $ANNEALING_CHECKPOINT $BASE_LUCIE_TOKENIZER_PATH $ANNEALING_OUTPUT
	i=$((i + 1))
done

# Stage 2
# for i in {1..4}; do
#     ANNEALING_CHECKPOINT=${BASE_CHECKPOINT_PATH}/stage2/mix_${i}/transformers_checkpoints/global_step1192
#     ANNEALING_OUTPUT=__lustre__fsn1__projects__rech__qgz__commun__trained_models__Lucie__stage2__mix_${i}__transformers_checkpoints__global_step1192
#     run_evaluation $ANNEALING_CHECKPOINT $BASE_LUCIE_TOKENIZER_PATH $ANNEALING_OUTPUT
# done
for i in {5..6}; do
    ANNEALING_CHECKPOINT=${BASE_CHECKPOINT_PATH}/stage2/mix_${i}/transformers_checkpoints/global_step1220
    ANNEALING_OUTPUT=__lustre__fsn1__projects__rech__qgz__commun__trained_models__Lucie__stage2__mix_${i}__transformers_checkpoints__global_step1220
    run_evaluation $ANNEALING_CHECKPOINT $BASE_LUCIE_TOKENIZER_PATH $ANNEALING_OUTPUT
done

# # Instruction
# INST_LUCIE_TOKENIZER_PATH=/lustre/fsn1/projects/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_65k_instruction_4kpad/tokenizer
# INST_OUTPUT='empty'

# INST_CHECKPOINT=${BASE_CHECKPOINT_PATH}/instruction/mix_1/transformers_checkpoints/global_step209
# run_evaluation $INST_CHECKPOINT $INST_LUCIE_TOKENIZER_PATH $INST_OUTPUT

# Instruction Assistant only
INST_LUCIE_TOKENIZER_PATH=/lustre/fsn1/projects/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_65k_instruction_4kpad/tokenizer
INST_OUTPUT='empty'

# INST_CHECKPOINT=${BASE_CHECKPOINT_PATH}/instruction_assistant_only/mix_1/transformers_checkpoints/global_step208
# run_evaluation $INST_CHECKPOINT $INST_LUCIE_TOKENIZER_PATH $INST_OUTPUT

# INST_CHECKPOINT=${BASE_CHECKPOINT_PATH}/instruction_assistant_only/mix_2/transformers_checkpoints/global_step525
# run_evaluation $INST_CHECKPOINT $INST_LUCIE_TOKENIZER_PATH $INST_OUTPUT

# INST_CHECKPOINT=${BASE_CHECKPOINT_PATH}/instruction_assistant_only/mix_3/transformers_checkpoints/global_step911
# run_evaluation $INST_CHECKPOINT $INST_LUCIE_TOKENIZER_PATH $INST_OUTPUT

# INST_CHECKPOINT=${BASE_CHECKPOINT_PATH}/instruction_assistant_only/mix_4/transformers_checkpoints/global_step317
# run_evaluation $INST_CHECKPOINT $INST_LUCIE_TOKENIZER_PATH $INST_OUTPUT

INST_CHECKPOINT=${BASE_CHECKPOINT_PATH}/instruction_assistant_only/mix_5/transformers_checkpoints/global_step615
run_evaluation $INST_CHECKPOINT $INST_LUCIE_TOKENIZER_PATH $INST_OUTPUT

# INST_CHECKPOINT=${BASE_CHECKPOINT_PATH}/instruction/mix_2/transformers_checkpoints/global_step526
# run_evaluation $INST_CHECKPOINT $INST_LUCIE_TOKENIZER_PATH $INST_OUTPUT

# INST_CHECKPOINT=${BASE_CHECKPOINT_PATH}/instruction/mix_3/transformers_checkpoints/global_step912
# run_evaluation $INST_CHECKPOINT $INST_LUCIE_TOKENIZER_PATH $INST_OUTPUT

# Evaluate instruction model with PEFT
# INSTRUCTION_CHECKPOINT=${BASE_CHECKPOINT_PATH}/pretrained/transformers_checkpoints/global_step753851
# INSTRUCTION_TOKENIZER_PATH=$ALL_CCFRSCRATCH/instruction_lora/Lucie/human/DemoCredi2Small_global_step753851__20241126_202052/checkpoint-final
# INSTRUCTION_OUTPUT=empty

# INSTRUCTION_PEFT_PATH=$ALL_CCFRSCRATCH/instruction_lora/Lucie/human/DemoCredi2Small_global_step753851__20241126_202052/checkpoint-final
# run_evaluation $INSTRUCTION_CHECKPOINT $INSTRUCTION_TOKENIZER_PATH $INSTRUCTION_OUTPUT $INSTRUCTION_PEFT_PATH

# INSTRUCTION_PEFT_PATH=$ALL_CCFRSCRATCH/instruction_lora/Lucie/human/DemoCredi2_global_step753851__20241126_202052/checkpoint-2700
# run_evaluation $INSTRUCTION_CHECKPOINT $INSTRUCTION_TOKENIZER_PATH $INSTRUCTION_OUTPUT $INSTRUCTION_PEFT_PATH

# INSTRUCTION_PEFT_PATH=$ALL_CCFRSCRATCH/instruction_lora/Lucie/human/DemoCredi2Small_v2_global_step753851/20241128_104748/checkpoint-final
# run_evaluation $INSTRUCTION_CHECKPOINT $INSTRUCTION_TOKENIZER_PATH $INSTRUCTION_OUTPUT $INSTRUCTION_PEFT_PATH '--apply_chat_template --fewshot_as_multiturn --log_samples'

# Wait for all background tasks to complete
wait