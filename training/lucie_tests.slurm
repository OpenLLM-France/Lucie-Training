#!/bin/bash
#SBATCH --job-name=test-lucie7b-1nodes-tp1-pp2-z0-b256-m1
#SBATCH --qos=qos_gpu-dev            # Avec ce qos la temps d'attente est moins long. Mais n'est possible qu'avec 32 GPUs maximum, donc à commenter si on lance sur plus de 32 GPUs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --constraint=a100
#SBATCH --time 00:15:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --account=qgz@a100

set -x -e

echo "START TIME: $(date)"

# Environement Set Up
module purge
module load cpuarch/amd 
module load anaconda-py3/2023.09 # Use this anaconda version already installed in Jean-Zay. If you're not on Jean-Zay, you have to install it.
module load cuda/12.1.0 # Use this cuda version already installed in Jean-Zay. If you're not on Jean-Zay, you have to install it.
module load gcc/12.2.0
conda activate lucie
# module load git/2.39.1

EXPE_NAME=testestest
OUTPUT_PATH=/gpfswork/rech/qgz/uzq54wg
CHECKPOINT_PATH=$OUTPUT_PATH/checkpoints/$EXPE_NAME
TENSORBOARD_PATH=$OUTPUT_PATH/tensorboard/$EXPE_NAME
LOGS_PATH=$OUTPUT_PATH/lucie-logs/$EXPE_NAME
mkdir -p $LOGS_PATH

MEGATRON_DEEPSPEED_REPO=/linkhome/rech/gendjf01/uzq54wg/Lucie-Training/Megatron-DeepSpeed
cd $MEGATRON_DEEPSPEED_REPO

DATASET=/gpfsscratch/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_2.4-space_prefix_all/Wikipedia--fr--026_text_document
TOKENIZER_PATH=OpenLLM-France/Lucie-tokenizer-v2.4-space_prefix_all

# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

GPUS_PER_NODE=8
NNODES=$SLURM_NNODES

TP=1
PP=2

GLOBAL_BATCH_SIZE=256
MICRO_BATCH_SIZE=1 # PP=2,TP=2: 6, PP=1,TP=2: 2, PP=2,TP=1: 2

# ----- model
HIDDEN_SIZE=4096 # e.g. llama-13b: 5120
FFN_HIDDEN_SIZE=11008 # e.g. llama-13b: 13824
NUM_LAYERS=32 # e.g. llama-13b: 40
NUM_HEADS=32 # e.g. llama-13b: 40
SEQ_LENGTH=4096 # for llama2 it's 4096
NUM_KV_HEADS=32 # llama2 70B uses GQA but for lucie and for now we use MHA, so NUM_KV_HEADS=NUM_HEADS

# ------ Optimizer
TRAIN_STEPS=250000 # e.g. llama: 1T tokens / 4M tokens_per_batch = 250000 steps
LR=3e-4
MIN_LR=3e-5
LR_WARMUP_STEPS=2000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

SAVE_INTERVAL=100

# ----- Deepspeed configuration
ZERO_STAGE=0

config_json="./ds_config.$SLURM_JOBID.json"

cat <<EOT > $config_json
{
  "train_batch_size" : $GLOBAL_BATCH_SIZE,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "steps_per_print": 1,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "bf16": {
    "enabled": true
  }
}
EOT

# no activation checkpoint
DEEPSPEED_ARGS=" \
    --deepspeed \
    --deepspeed_config ${config_json} \
    --zero-stage ${ZERO_STAGE} \
    "

# activation checkpoint
# DEEPSPEED_ARGS=" \
#     --deepspeed \
#     --deepspeed_config ${config_json} \
#     --zero-stage ${ZERO_STAGE} \
#     --deepspeed-activation-checkpointing \
#     --recompute-granularity full \
#     --recompute-method uniform \
#     "

LOG_ARGS=" \
       --log-interval 1 \
       --tensorboard-dir $TENSORBOARD_PATH \
       --tensorboard-log-interval 1 \
       --log-timers-to-tensorboard \
       --log-batch-size-to-tensorboard \
       --log-memory-to-tensorboard \
       --log-world-size-to-tensorboard \
       --tensorboard-queue-size 5 \
       "

# do not remove or the training will hang and nodes will be lost w/o this workaround
export CUDA_LAUNCH_BLOCKING=1

# hide duplicated errors using this hack - will be properly fixed in pt-1.12
export TORCHELASTIC_ERROR_FILE=/tmp/torch-elastic-error.json

# --pipeline-model-parallel-size $PP \
# --no-pipeline-parallel \
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank \$SLURM_PROCID --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT --rdzv_backend c10d --max_restarts 0 --tee 3"
export RUN="torchrun $DISTRIBUTED_ARGS \
       `pwd`/pretrain_gpt.py \
       --data-cache-path /gpfswork/rech/qgz/uzq54wg/.cache \
       --tensor-model-parallel-size $TP \
       --pipeline-model-parallel-size $PP \
       --num-layers $NUM_LAYERS \
       --hidden-size $HIDDEN_SIZE \
       --ffn-hidden-size $FFN_HIDDEN_SIZE \
       --num-attention-heads $NUM_HEADS \
       --micro-batch-size $MICRO_BATCH_SIZE \
       --global-batch-size $GLOBAL_BATCH_SIZE \
       --seq-length $SEQ_LENGTH \
       --max-position-embeddings $SEQ_LENGTH \
       --train-iters $TRAIN_STEPS \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATASET \
       --data-impl mmap \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path $TOKENIZER_PATH \
       --distributed-backend nccl \
       --lr $LR \
       --lr-decay-style cosine \
       --min-lr $MIN_LR \
       --weight-decay $WEIGHT_DECAY \
       --clip-grad $GRAD_CLIP \
       --lr-warmup-iters $LR_WARMUP_STEPS \
       --optimizer adam \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --save-interval 100 \
       --eval-interval 100 \
       --eval-iters 0 \
       --split 1,0,0 \
       --bf16 \
       --use-flash-attn-v2 \
       --no-query-key-layer-scaling \
       --attention-dropout 0 \
       --hidden-dropout 0 \
       --use-rotary-position-embeddings \
       --untie-embeddings-and-output-weights \
       --swiglu \
       --normalization rmsnorm \
       --disable-bias-linear \
       --num-key-value-heads $NUM_KV_HEADS \
       $DEEPSPEED_ARGS \
       $LOG_ARGS \
       $EXIT_OPTS \
       "

srun --jobid $SLURM_JOBID bash -c "$RUN" 2>&1 | tee -a $LOGS_PATH/main_log.txt
