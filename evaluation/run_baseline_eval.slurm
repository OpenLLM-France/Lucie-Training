#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8                  # Number of tasks
#SBATCH --gpus-per-task=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=8           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --constraint=a100            # pour cibler les noeuds H100
#SBATCH --account=qgz@a100
#SBATCH --qos=qos_gpu-dev
#SBATCH --time 2:00:00               # maximum execution time (HH:MM:SS)
#SBATCH --output=slurm_logs/%x-%j.out           # output file name
#SBATCH --job-name=baseline_eval

#### English benchmarks:
#  mmlu,
#  arc_challenge,
#  hellaswag,
#  winogrande,
#  openbookqa,
#  piqa

#### French benchmarks:
# m_mmlu_fr,
# french_bench_arc_challenge,
# french_bench_hellaswag,
# xwinograd_fr,
# french_bench_multifquad,
# french_bench_fquadv2_genq,
# french_bench_fquadv2_hasAns,
# french_bench_grammar,
# french_bench_vocab,
# belebele_fra_Latn

source set_env.sh

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1 
export HF_HUB_OFFLINE=1

run_eval(){
    local MODEL_NAME=$1
    local SHOTS=$2
    local TASKS=$3

    lm_eval --model hf \
        --model_args pretrained=${MODEL_NAME} \
        --tasks ${TASKS} \
        --num_fewshot ${SHOTS} \
        --batch_size auto \
        --output_path out \
        --seed 42
}

run_eval_pythia(){
    local MODEL_NAME=$1
    local STEP=$2
    local SHOTS=$3
    local TASKS=$4

    lm_eval --model hf \
        --model_args pretrained=${MODEL_NAME},revision=step${STEP} \
        --tasks ${TASKS} \
        --num_fewshot ${SHOTS} \
        --batch_size auto \
        --output_path out \
        --seed 42
}

# Export the function so it can be used by srun
export -f run_eval
export -f run_eval_pythia

export TASKS=mmlu,arc_challenge,arc_easy,openbookqa,m_mmlu_fr
export DIR=/gpfsdswork/dataset/HuggingFace_Models

# Run the function in parallel using srun
# srun --exclusive --ntasks=1 bash -c 'run_eval "$DIR/tiiuae/falcon-7b" 5 "$TASKS"' &
# srun --exclusive --ntasks=1 bash -c 'run_eval "$DIR/meta-llama/Meta-Llama-3-8B" 5 "$TASKS"' &
# srun --exclusive --ntasks=1 bash -c 'run_eval "$DIR/mistralai/Mistral-7B-v0.1" 5 "$TASKS"' &
# srun --exclusive --ntasks=1 bash -c 'run_eval EleutherAI/pythia-6.9b 5 "$TASKS"' &
srun --exclusive --ntasks=1 bash -c 'run_eval EleutherAI/pythia-6.9b 5 "$TASKS"' &
srun --exclusive --ntasks=1 bash -c 'run_eval_pythia EleutherAI/pythia-6.9b 100000 5 "$TASKS"' &
srun --exclusive --ntasks=1 bash -c 'run_eval_pythia EleutherAI/pythia-6.9b 50000 5 "$TASKS"' &

# Wait for all tasks to finish
wait