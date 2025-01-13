#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=64           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:8                 # number of gpus per nodes
#SBATCH --constraint=a100
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=./out/%x-%j.out # STDOUT
#SBATCH --account=xxx@a100
#SBATCH --qos=qos_gpu-t3
#SBATCH --job-name=lucie-test
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ogouvert@linagora.com
#SBATCH --job-name=lucie7b-ablation-study

set -x -e

echo "START TIME: $(date)"

# Load env and variables
source ~/Lucie-Training/training/set_env.sh

VARIANT=ablation_study
CHECKPOINT_PATH=$OUTPUT_PATH/checkpoints/$VARIANT
TENSORBOARD_PATH=$OUTPUT_PATH/tensorboard/$VARIANT/$SLURM_JOB_NAME
LOGS_PATH=$OUTPUT_PATH/lucie-logs/$VARIANT/$SLURM_JOB_NAME
mkdir -p $LOGS_PATH

cd $MEGATRON_DEEPSPEED_REPO

TOKENIZER_PATH=OpenLLM-France/Lucie-tokenizer-65k
TOKENS_DIR="/gpfsssd/scratch/rech/xxx/commun/preprocessed_data/Lucie/lucie_tokens_65k_grouped"
DATASET="$(python ~/Lucie-Training/training/collect_data_and_weights.py $TOKENS_DIR --wikipedia_weight 1)"
DATA_CACHE_PATH=/gpfswork/rech/xxx/xxx00xx/.cache

# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

GPUS_PER_NODE=8
NNODES=$SLURM_NNODES

# ----- model
HIDDEN_SIZE=512 
FFN_HIDDEN_SIZE=1536 
NUM_LAYERS=6 
NUM_HEADS=8
NUM_KV_HEADS=8 
SEQ_LENGTH=4096

GLOBAL_BATCH_SIZE=256
MICRO_BATCH_SIZE=16
ZERO_STAGE=0

# ------ Optimizer
TRAIN_STEPS=30_000 
LR=3e-4
MIN_LR=3e-5
LR_WARMUP_STEPS=2000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

SAVE_INTERVAL=10_000

config_json="./ds_config.$SLURM_JOBID.json"

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
  "steps_per_print": 4000,
  "wall_clock_breakdown": false
}
EOT

DEEPSPEED_ARGS=" \
    --deepspeed \
    --deepspeed_config ${config_json} \
    --zero-stage ${ZERO_STAGE} \
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
       --clip-grad $GRAD_CLIP \
       --lr-warmup-iters $LR_WARMUP_STEPS \
       --optimizer adam \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --adam-eps 1e-5 \
       --weight-decay $WEIGHT_DECAY \
       "

LOG_ARGS=" \
       --log-interval 1 \
       --tensorboard-dir $TENSORBOARD_PATH \
       --tensorboard-log-interval 200 \
       --log-timers-to-tensorboard \
       --log-batch-size-to-tensorboard \
       --log-memory-to-tensorboard \
       --log-world-size-to-tensorboard \
       --tensorboard-queue-size 1 \
       --no-log-loss-scale-to-tensorboard \
       "

# do not remove or the training will hang and nodes will be lost w/o this workaround
export CUDA_LAUNCH_BLOCKING=1

# hide duplicated errors using this hack - will be properly fixed in pt-1.12
export TORCHELASTIC_ERROR_FILE=/tmp/torch-elastic-error.json

DISTRIBUTED_ARGS=" \
       --nproc_per_node $GPUS_PER_NODE \
       --nnodes $NNODES \
       --node_rank \$SLURM_PROCID \
       --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
       --rdzv_backend c10d \
       --max_restarts 0 \
       --tee 3 \
       "

EXIT_OPTS=" \
    --exit-duration-in-mins 1190 \
    "

export RUN="torchrun $DISTRIBUTED_ARGS \
       `pwd`/pretrain_gpt.py \
       --data-cache-path $DATA_CACHE_PATH \
       --tensor-model-parallel-size 1 \
       --pipeline-model-parallel-size 1 \
       --micro-batch-size $MICRO_BATCH_SIZE \
       --global-batch-size $GLOBAL_BATCH_SIZE \
       --train-iters $TRAIN_STEPS \
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
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       $GPT_ARGS \
       $OPTIMIZER_ARGS \
       $DEEPSPEED_ARGS \
       $LOG_ARGS \
       $EXIT_OPTS \
       "

clear; srun --jobid $SLURM_JOBID bash -c "$RUN" 2>&1 | tee -a $LOGS_PATH/main_log.txt

echo "END TIME: $(date)"