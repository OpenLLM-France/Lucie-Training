# ----- parallelism
SLURM_JOB_NAME="test_perplexity"
SLURM_NNODES=1
TP=1
PP=1

if [ $PP -eq 0 ]; then
    EXTRA_ARGS="--no-pipeline-parallel"
    PP=1
else
    EXTRA_ARGS=""
fi

MASTER_ADDR=127.0.0.1 #$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

GPUS_PER_NODE=2
NNODES=$SLURM_NNODES

# ----- data
TOKENIZER_PATH=/datasets/lucie_tokens_65k_grouped/tokenizer
TOKENS_DIR=/local_data/lucie_tokens_65k_grouped
DATASET="$(python ~/Lucie-Training/training/collect_data_and_weights_alt.py $TOKENS_DIR)"

if [ -z "$DATASET" ]; then
  echo "No data found"
  exit 1
fi

DATA_CACHE_PATH=results/test_datasets2

# ----- results
PERPLEXITY_RESULTS_PATH=results/$SLURM_JOB_NAME

# ----- model
HIDDEN_SIZE=512 # e.g. llama-13b: 5120
FFN_HIDDEN_SIZE=1536 # e.g. llama-13b: 13824
NUM_LAYERS=6 # e.g. llama-13b: 40
NUM_HEADS=8 # e.g. llama-13b: 40
SEQ_LENGTH=4096 # for llama2 it's 4096
NUM_KV_HEADS=4 # llama2 70B uses GQA but for lucie and for now we use MHA, so NUM_KV_HEADS=NUM_HEADS

ZERO_STAGE=0
GLOBAL_BATCH_SIZE=16
MICRO_BATCH_SIZE=4
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
  "steps_per_print": 4000,
  "wall_clock_breakdown": false
}
EOT

CHECKPOINT_PATH=/home/lucas.hervier/Lucie-Training/results/checkpoints/test0

# ------ Optimizer
TRAIN_STEPS=250000 # e.g. llama: 1T tokens / 4M tokens_per_batch = 250000 steps
LR=3e-4
MIN_LR=3e-5
LR_WARMUP_STEPS=2000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

SAVE_INTERVAL=1000

OPTIMIZER_ARGS=" \
       --lr $LR \
       --lr-decay-style cosine \
       --min-lr $MIN_LR \
       --weight-decay $WEIGHT_DECAY \
       --clip-grad $GRAD_CLIP \
       --lr-warmup-iters $LR_WARMUP_STEPS \
       --optimizer adam \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       "

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

# do not remove or the training will hang and nodes will be lost w/o this workaround
export CUDA_LAUNCH_BLOCKING=1

# hide duplicated errors using this hack - will be properly fixed in pt-1.12
export TORCHELASTIC_ERROR_FILE=/tmp/torch-elastic-error.json

DISTRIBUTED_ARGS=" \
       --nproc_per_node $GPUS_PER_NODE \
       --nnodes $NNODES \
       --node_rank 0 \
       --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
       --rdzv_backend c10d \
       --max_restarts 0 \
       --tee 3 \
       "

torchrun $DISTRIBUTED_ARGS \
      `pwd`/perplexity.py \
      --data-path $DATASET \
      --tensor-model-parallel-size $TP \
      --pipeline-model-parallel-size $PP \
      $EXTRA_ARGS \
      --data-impl mmap \
      --split 0.9,0.05,0.05 \
      --seed 42 \
      --skip-warmup True\
      --micro-batch-size 4 \
      --datatest-cache-path $DATA_CACHE_PATH \
      $GPT_ARGS \
      $OPTIMIZER_ARGS \
      --tokenizer-type PretrainedFromHF  \
      --tokenizer-name-or-path $TOKENIZER_PATH \
      --distributed-backend nccl \
      --load $CHECKPOINT_PATH \
      --load-iteration 3000 \
      --inference \
      --finetune \
      $DEEPSPEED_ARGS \
      --perplexity-results-path $PERPLEXITY_RESULTS_PATH \
