#!/bin/bash
#SBATCH --nodes=128
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=96           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --constraint=h100                      # pour cibler les noeuds H100
#SBATCH --gres=gpu:4                 # nombre de GPU par noeud (max 4 pour les noeuds H100)
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --job-name=lucie-pretraining
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ogouvert@linagora.com

set -x -e

echo "START TIME: $(date)"

export MEGATRON_DEEPSPEED_REPO=/linkhome/rech/gendjf01/uzq54wg/Lucie-Training/Megatron-DeepSpeed
export SRC_DIR=~/Lucie-Training

# Load env and variables
source $SRC_DIR/training/set_env.sh

# Variables
CHECKPOINT_PATH=$ALL_CCFRSCRATCH/trained_models/Lucie/pretrained/checkpoints
TENSORBOARD_PATH=$ALL_CCFRSCRATCH/trained_models/Lucie/pretrained/tensorboard/pretraining
LOGS_PATH=$ALL_CCFRSCRATCH/trained_models/Lucie/pretrained/logs
mkdir -p $LOGS_PATH

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
TOKENS_DIR="/lustre/fsn1/projects/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_65k_grouped"
TOKENIZER_PATH=$TOKENS_DIR/tokenizer 
DATASET="$(python $SRC_DIR/training/collect_data_and_weights_light.py $TOKENS_DIR)"

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
SEQ_LENGTH=4096 
NUM_KV_HEADS=8 

GLOBAL_BATCH_SIZE=1024
GLOBAL_BATCH_SIZE_MIN=256
GLOBAL_BATCH_SIZE_STEP=64
MICRO_BATCH_SIZE=2
TP=4
PP=4
ZERO_STAGE=0

# ------ Optimizer
LR=3e-4
MIN_LR=3e-5
WEIGHT_DECAY=0.1
GRAD_CLIP=1

TRAIN_SAMPLES=762_144_586 # 3T tokens
LR_DECAY_SAMPLES=762_144_586 # 3T tokens - not used right now
LR_WARMUP_SAMPLES=2_000_000
RAMPUP_SAMPLES=10_000_000

SAVE_INTERVAL=5000

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
  "gradient_clipping": 1.0,
  "steps_per_print": 2000,
  "wall_clock_breakdown": false
}
EOT

DEEPSPEED_ARGS=" \
    --deepspeed \
    --deepspeed_config ${config_json} \
    --zero-stage ${ZERO_STAGE} \
    --deepspeed-activation-checkpointing \
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
       --untie-embeddings-and-output-weights \
       --swiglu \
       --normalization rmsnorm \
       --disable-bias-linear \
       --num-key-value-heads $NUM_KV_HEADS \
       --bf16 \
       "

OPTIMIZER_ARGS=" \
       --lr $LR \
       --lr-decay-style cosine \
       --min-lr $MIN_LR \
       --lr-warmup-samples $LR_WARMUP_SAMPLES \
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

# for 20h 1180, i.e., 20min for last checkpoint
EXIT_OPTS=" \
    --exit-duration-in-mins 1180 \ 
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

DATA_CACHE_PATH=$TOKENS_DIR"/data_indices_ts-"$TRAIN_SAMPLES"_sl-"$SEQ_LENGTH

export RUN=" torchrun $DISTRIBUTED_ARGS \
       `pwd`/pretrain_gpt.py \
       --data-cache-path $DATA_CACHE_PATH \
       --tensor-model-parallel-size $TP \
       --pipeline-model-parallel-size $PP \
       --micro-batch-size $MICRO_BATCH_SIZE \
       --global-batch-size $GLOBAL_BATCH_SIZE \
       --train-samples $TRAIN_SAMPLES \
       --rampup-batch-size $GLOBAL_BATCH_SIZE_MIN $GLOBAL_BATCH_SIZE_STEP $RAMPUP_SAMPLES \
       --data-path $DATASET \
       --data-impl mmap \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path $TOKENIZER_PATH \
       --distributed-backend nccl \
       --save-interval $SAVE_INTERVAL \
       --eval-interval 1000 \
       --eval-iters 0 \
       --split 1,0,0 \
       --use-flash-attn-v2 \
       --no-query-key-layer-scaling \
       --seed 42 \
       --init-method-std 0.009 \
       --checkpoint-activations \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       $GPT_ARGS \
       $OPTIMIZER_ARGS \
       $DEEPSPEED_ARGS \
       $LOG_ARGS \
       $EXIT_OPTS \
       "

SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

clear; srun $SRUN_ARGS --jobid $SLURM_JOBID bash -c "$RUN" 2>&1 | tee -a $LOGS_PATH/main_log.txt

echo "END TIME: $(date)"