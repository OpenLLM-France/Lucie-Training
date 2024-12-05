#!/bin/bash

# Load necessary modules and activate environment
module purge
module load arch/h100
module load anaconda-py3/2024.06
conda activate $SCRATCH/envs/evaluation

# Paths
BASE_CHECKPOINT_PATH=$ALL_CCFRSCRATCH/trained_models/Lucie
BASE_OUTPUT_PATH=out/french_bench_gen/lucie
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
            --tasks french_bench_gen \
            --num_fewshot 5 \
            --batch_size auto \
            --output_path $BASE_OUTPUT_PATH &
    fi
}

# Evaluate the final checkpoint
run_evaluation "${BASE_CHECKPOINT_PATH}/pretrained/transformers_checkpoints/global_step753851" \
    $BASE_LUCIE_TOKENIZER_PATH "${OUTPUT_PREFIX}753851"

# Stage 2
ANNEALING_CHECKPOINT=${BASE_CHECKPOINT_PATH}/stage2/transformers_checkpoints/global_step1192
ANNEALING_OUTPUT=__lustre__fsn1__projects__rech__qgz__commun__trained_models__Lucie__stage2__transformers_checkpoints__global_step1192
ANNEALING_OUTPUT='empty'
run_evaluation $ANNEALING_CHECKPOINT $BASE_LUCIE_TOKENIZER_PATH $ANNEALING_OUTPUT

wait