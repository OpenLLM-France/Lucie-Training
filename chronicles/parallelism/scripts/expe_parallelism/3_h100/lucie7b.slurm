#!/bin/bash
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=96           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH -C h100                      # pour cibler les noeuds H100
#SBATCH --gres=gpu:4                 # nombre de GPU par noeud (max 4 pour les noeuds H100)
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --account=xxx@h100
#SBATCH --qos=qos_gpu-t3
#SBATCH --job-name=lucie-test
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ogouvert@linagora.com

set -x -e

echo "START TIME: $(date)"

echo "TP: $TP"
echo "PP: $PP"
echo "GLOBAL_BATCH_SIZE: $GLOBAL_BATCH_SIZE"
echo "MICRO_BATCH_SIZE: $MICRO_BATCH_SIZE"

# Load env and variables
source ~/Lucie-Training/training/h100_tests/set_env.sh
cd $MEGATRON_DEEPSPEED_REPO


OUTPUT_PATH=$SCRATCH/expe_z0
CHECKPOINT_PATH=$OUTPUT_PATH/checkpoints
TENSORBOARD_PATH=$OUTPUT_PATH/tensorboard

# Tokenizer and data
TOKENIZER_PATH=OpenLLM-France/Lucie-tokenizer-65k
TOKENS_DIR="/gpfsssd/scratch/rech/xxx/commun/preprocessed_data/Lucie/lucie_tokens_65k_grouped"
DATASET="$(python ~/Lucie-Training/training/collect_data_and_weights.py $TOKENS_DIR)"

if [ -z "$DATASET" ]; then
  echo "No data found"
  exit 1
fi

# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

# ----- model
HIDDEN_SIZE=4096 
FFN_HIDDEN_SIZE=12288 
NUM_LAYERS=32 
NUM_HEADS=32
SEQ_LENGTH=4096 
NUM_KV_HEADS=8 

# ------ Optimizer
LR=3e-4
MIN_LR=3e-5
WEIGHT_DECAY=0.1
GRAD_CLIP=1

TRAIN_SAMPLES=750_000_000 # 3T tokens
LR_DECAY_SAMPLES=750_000_000 # 3T tokens - not used right now
LR_WARMUP_SAMPLES=2_000_000

SAVE_INTERVAL=1000

config_json="./ds_config.json"

if [ $ZERO_STAGE -eq 0 ]; then
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
else
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
fi

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

# for 20h 1190
EXIT_OPTS=" \
       --exit-interval 3 \
    "

# do not remove or the training will hang and nodes will be lost w/o this workaround
export CUDA_LAUNCH_BLOCKING=1

# hide duplicated errors using this hack - will be properly fixed in pt-1.12
export TORCHELASTIC_ERROR_FILE=/tmp/torch-elastic-error.json

DISTRIBUTED_ARGS=" \
       --nproc_per_node $SBATCH_GPUS_PER_NODE \
       --nnodes $SLURM_NNODES \
       --node_rank \$SLURM_PROCID \
       --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
       --rdzv_backend c10d \
       --max_restarts 0 \
       --tee 3 \
       "

DATA_CACHE_PATH=$LOGS_PATH/data_indices

export RUN="torchrun $DISTRIBUTED_ARGS \
       `pwd`/pretrain_gpt.py \
       --data-cache-path $DATA_CACHE_PATH \
       --tensor-model-parallel-size $TP \
       --pipeline-model-parallel-size $PP \
       --micro-batch-size $MICRO_BATCH_SIZE \
       --global-batch-size $GLOBAL_BATCH_SIZE \
       --train-samples $TRAIN_SAMPLES \
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
       --save $CHECKPOINT_PATH \
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

clear; srun $SRUN_ARGS --jobid $SLURM_JOBID bash -c "$RUN"

echo "END TIME: $(date)"