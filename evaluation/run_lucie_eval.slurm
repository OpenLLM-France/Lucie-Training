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
#SBATCH --job-name=lucie_eval

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
    local CHECKPOINT_STEP=$1
    local SHOTS=$2
    local TASKS=$3

    CHECKPOINT_PATH=$ALL_CCFRSCRATCH/trained_models/Lucie/pretrained/transformers_checkpoints/global_step${CHECKPOINT_STEP}
    TOKENIZER_PATH=/lustre/fsn1/projects/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_65k_grouped/tokenizer

    lm_eval --model hf \
        --model_args pretrained=${CHECKPOINT_PATH},tokenizer=${TOKENIZER_PATH},dtype=bfloat16 \
        --tasks ${TASKS} \
        --num_fewshot ${SHOTS} \
        --batch_size auto \
        --output_path out \
        --seed 42
}

# Export the function so it can be used by srun
export -f run_eval

export TASKS=mmlu,arc_challenge,arc_easy,openbookqa,m_mmlu_fr
export i=25000

while [ $i -le 240000 ]; do
    srun --exclusive --ntasks=1 bash -c 'run_eval "$i" 5 "$TASKS"' &
    i=$((i + 25000))
done

wait