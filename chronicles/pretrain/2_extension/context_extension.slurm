#!/bin/bash
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=96           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH -A xxx@h100                  # comptabilite a utiliser, avec xyz le trigramme de votre projet
#SBATCH --constraint=h100                      # pour cibler les noeuds H100
#SBATCH --gres=gpu:4                 # nombre de GPU par noeud (max 4 pour les noeuds H100)
#SBATCH --time=20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=lucie_slurm_logs/%x-%j.out           # output file name
#SBATCH --job-name=context_extension
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ogouvert@linagora.com,jlouradour@linagora.com

set -x -e

echo "START TIME: $(date)"

ZERO_STAGE=1
SEQ_LENGTH=32_000 
GLOBAL_BATCH_SIZE=128
MICRO_BATCH_SIZE=4
TP=4
PP=4

export MEGATRON_DEEPSPEED_REPO=/linkhome/rech/xxxxx00/xxx00xx/Lucie-Training/Megatron-DeepSpeed-sandbox
export SRC_DIR=~/Lucie-Training

# Load env and variables
source $SRC_DIR/training/set_env.sh

# Variables
UNIVERSAL_CHECKPOINT_PATH=$ALL_CCFRSCRATCH/trained_models/Lucie/pretrained/universal_checkpoints
SAVE_CHECKPOINT_PATH=$ALL_CCFRSCRATCH/trained_models/Lucie/extension_rope20M/checkpoints
TENSORBOARD_PATH=$ALL_CCFRSCRATCH/trained_models/Lucie/extension_rope20M/tensorboard/pretraining
LOGS_PATH=$ALL_CCFRSCRATCH/trained_models/Lucie/extension_rope20M/logs
mkdir -p $LOGS_PATH

echo global_step753851 > /lustre/fsn1/projects/rech/xxx/commun/trained_models/Lucie/pretrained/universal_checkpoints/latest_universal

export TORCH_CUDA_ARCH_LIST="9.0+PTX" 
export OMP_NUM_THREADS=1
# do not remove or the training will hang and nodes will be lost w/o this workaround
export CUDA_LAUNCH_BLOCKING=1
# hide duplicated errors using this hack - will be properly fixed in pt-1.12
export TORCHELASTIC_ERROR_FILE=/tmp/torch-elastic-error.json
# force crashing on nccl issues like hanging broadcast
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd $MEGATRON_DEEPSPEED_REPO # "megatron" will be in PYTHONPATH (current folder)

# Tokenizer and data
TOKENS_DIR=/lustre/fsn1/projects/rech/xxx/commun/preprocessed_data/Lucie/lucie_tokens_65k_grouped
DATASET="$(python $SRC_DIR/training/collect_data_and_weights_extension.py)"
TOKENIZER_PATH=$TOKENS_DIR/tokenizer

if [ -z "$DATASET" ]; then
  echo "No data found"
  exit 1
fi

# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

GPUS_PER_NODE=4

# ----- model
HIDDEN_SIZE=4096 
FFN_HIDDEN_SIZE=12288 
NUM_LAYERS=32 
NUM_HEADS=32
NUM_KV_HEADS=8 
ROPE_THETA=20_000_000

# ------ Optimizer
LR=2e-5
WEIGHT_DECAY=0.1
GRAD_CLIP=1

TRAIN_SAMPLES=156250 # 5B tokens

SAVE_INTERVAL=250

config_json="$LOGS_PATH/ds_config.json"

cat <<EOT > $config_json
{
  "train_batch_size" : $GLOBAL_BATCH_SIZE,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "bf16": {
    "enabled": true
  },
  "data_types": {
    "grad_accum_dtype": "fp32" 
  },
  "gradient_clipping": 1.0,
  "steps_per_print": 2000,
  "wall_clock_breakdown": false
}
EOT

DEEPSPEED_ARGS=" \
    --deepspeed \
    --deepspeed_config ${config_json} \
    --zero-stage ${ZERO_STAGE} \
    --recompute-granularity full \
    --recompute-method uniform \
    "

GPT_ARGS=" \
       --num-layers $NUM_LAYERS \
       --hidden-size $HIDDEN_SIZE \
       --ffn-hidden-size $FFN_HIDDEN_SIZE \
       --num-attention-heads $NUM_HEADS \
       --seq-length $SEQ_LENGTH \
       --max-position-embeddings $SEQ_LENGTH \
       --attention-dropout 0 \
       --hidden-dropout 0 \
       --use-rotary-position-embeddings \
       --rotary-position-embeddings-theta $ROPE_THETA \
       --untie-embeddings-and-output-weights \
       --swiglu \
       --normalization rmsnorm \
       --disable-bias-linear \
       --num-key-value-heads $NUM_KV_HEADS \
       --bf16 \
       "

OPTIMIZER_ARGS=" \
       --lr $LR \
       --lr-decay-style constant \
       --clip-grad $GRAD_CLIP \
       --optimizer adam \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --adam-eps 1e-5 \
       --weight-decay $WEIGHT_DECAY \
       "

LOG_ARGS=" \
       --log-interval 1 \
       --tensorboard-dir $TENSORBOARD_PATH \
       --tensorboard-log-interval 1 \
       --log-timers-to-tensorboard \
       --log-batch-size-to-tensorboard \
       --log-memory-to-tensorboard \
       --log-world-size-to-tensorboard \
       --tensorboard-queue-size 5 \
       --no-log-loss-scale-to-tensorboard \
       "

EXIT_OPTS=" \
    --exit-duration-in-mins 1190 \
    "

DISTRIBUTED_ARGS=" \
       --nproc_per_node $GPUS_PER_NODE \
       --nnodes $SLURM_NNODES \
       --node_rank \$SLURM_PROCID \
       --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
       --rdzv_backend c10d \
       --max_restarts 0 \
       --tee 3 \
       "

DATA_CACHE_PATH=${TOKENS_DIR}/extension/data_indices_ts-${TRAIN_SAMPLES}_sl-${SEQ_LENGTH}

export RUN=" torchrun $DISTRIBUTED_ARGS \
       `pwd`/pretrain_gpt.py \
       --data-cache-path $DATA_CACHE_PATH \
       --tensor-model-parallel-size $TP \
       --pipeline-model-parallel-size $PP \
       --micro-batch-size $MICRO_BATCH_SIZE \
       --global-batch-size $GLOBAL_BATCH_SIZE \
       --train-samples $TRAIN_SAMPLES \
       --data-path $DATASET \
       --data-impl mmap \
       --tokenizer-type PretrainedFromHF \
       --tokenizer-name-or-path $TOKENIZER_PATH \
       --distributed-backend nccl \
       --save-interval $SAVE_INTERVAL \
       --eval-interval 1000 \
       --eval-iters 0 \
       --split 1,0,0 \
       --no-query-key-layer-scaling \
       --use-flash-attn-v2 \
       --seed 42 \
       --load $UNIVERSAL_CHECKPOINT_PATH \
       --save $SAVE_CHECKPOINT_PATH \
       $GPT_ARGS \
       $OPTIMIZER_ARGS \
       $LOG_ARGS \
       $DEEPSPEED_ARGS \
       $EXIT_OPTS \
       --universal-checkpoint \
       --finetune \
       "

SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

clear; srun $SRUN_ARGS --jobid $SLURM_JOBID bash -c "$RUN" 2>&1 | tee -a $LOGS_PATH/main_log.txt

echo "END TIME: $(date)"