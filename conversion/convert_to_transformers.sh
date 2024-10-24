#!/bin/bash

export MEGATRON_DEEPSPEED_REPO=/linkhome/rech/gendjf01/uzq54wg/Lucie-Training/Megatron-DeepSpeed-sandbox
export PYTHONPATH=$MEGATRON_DEEPSPEED_REPO
cd $MEGATRON_DEEPSPEED_REPO

run_conversion(){
    local MAIN_PATH=$1
    local GLOBAL_STEP=$2

    MEGATRON_CHECKPOINT_PATH=$MAIN_PATH/checkpoints/global_step${GLOBAL_STEP}
    UNIVERSAL_CHECKPOINT_PATH=$MAIN_PATH/universal_checkpoints/global_step${GLOBAL_STEP}
    TRANSFORMERS_CHECKPOINT_PATH=$MAIN_PATH/transformers_checkpoints/global_step${GLOBAL_STEP}

    if [ ! -d $UNIVERSAL_CHECKPOINT_PATH ]; then
        # DS to Universal
        srun --exclusive -n 1 python tools/convert_checkpoint/ds_to_universal.py --input_folder $MEGATRON_CHECKPOINT_PATH --output_folder $UNIVERSAL_CHECKPOINT_PATH 
    fi

    # Universal to Transformer
    
    if [ ! -d $TRANSFORMERS_CHECKPOINT_PATH ]; then
    # if [ ! -d $TRANSFORMERS_CHECKPOINT_PATH/pytorch_model.bin.index.json ]; then
        srun --exclusive -n 1 python tools/convert_checkpoint/universal_to_hf_llama.py --input_folder $UNIVERSAL_CHECKPOINT_PATH --output_folder $TRANSFORMERS_CHECKPOINT_PATH --max_shard_size 5GB 
    fi
}

### LUCIE PRETRAINED
each 5000
export i=5000
while [ $i -le 20000 ]; do
    run_conversion $ALL_CCFRSCRATCH/trained_models/Lucie/pretrained $i &
    i=$((i + 5000))
done

# each 25000
export i=25000
while [ $i -le 660000 ]; do
    run_conversion $ALL_CCFRSCRATCH/trained_models/Lucie/pretrained $i &
    i=$((i + 25000))
done

# run_conversion $ALL_CCFRSCRATCH/trained_models/Lucie/pretrained 485000 &

wait