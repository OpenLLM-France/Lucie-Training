#!/bin/bash

MEGATRON_DEEPSPEED_REPO=/linkhome/rech/gendjf01/uzq54wg/Lucie-Training/Megatron-DeepSpeed-sandbox
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
i=5000
while [ $i -le 20000 ]; do
    run_conversion $ALL_CCFRSCRATCH/trained_models/Lucie/pretrained $i &
    i=$((i + 5000))
done

# each 25000
i=25000
while [ $i -le 750000 ]; do
    run_conversion $ALL_CCFRSCRATCH/trained_models/Lucie/pretrained $i &
    i=$((i + 25000))
done

# Final checkpoint
run_conversion $ALL_CCFRSCRATCH/trained_models/Lucie/pretrained 753851 &

# Extension
i=250
while [ $i -le 1000 ]; do
    run_conversion $ALL_CCFRSCRATCH/trained_models/Lucie/extension_rope20M $i &
    i=$((i + 250))
done
run_conversion $ALL_CCFRSCRATCH/trained_models/Lucie/extension_rope20M 1220 &

# Annealing
global_step=1
while [ $i -le 9 ]; do
    run_conversion $ALL_CCFRSCRATCH/trained_models/Lucie/annealing/mix_1 $global_step &
    run_conversion $ALL_CCFRSCRATCH/trained_models/Lucie/annealing/mix_2 $global_step &
    run_conversion $ALL_CCFRSCRATCH/trained_models/Lucie/annealing/mix_3 $global_step &
    i=$((i + 1))
done

run_conversion $ALL_CCFRSCRATCH/trained_models/Lucie/annealing/mix_4 9 &

# Stage 2...
run_conversion $ALL_CCFRSCRATCH/trained_models/Lucie/stage2/mix_1 1192 &
run_conversion $ALL_CCFRSCRATCH/trained_models/Lucie/stage2/mix_2 1192 &
run_conversion $ALL_CCFRSCRATCH/trained_models/Lucie/stage2/mix_3 1192 &
run_conversion $ALL_CCFRSCRATCH/trained_models/Lucie/stage2/mix_4 1192 &
run_conversion $ALL_CCFRSCRATCH/trained_models/Lucie/stage2/mix_5 1220 &
run_conversion $ALL_CCFRSCRATCH/trained_models/Lucie/stage2/mix_6 1220 &

# Instruction
run_conversion $ALL_CCFRSCRATCH/trained_models/Lucie/instruction/mix_1 209 &
run_conversion $ALL_CCFRSCRATCH/trained_models/Lucie/instruction/mix_2 526 &
run_conversion $ALL_CCFRSCRATCH/trained_models/Lucie/instruction/mix_3 912 &

# Instruction
run_conversion $ALL_CCFRSCRATCH/trained_models/Lucie/instruction_assistant_only/mix_1 208 &
run_conversion $ALL_CCFRSCRATCH/trained_models/Lucie/instruction_assistant_only/mix_2 525 &
run_conversion $ALL_CCFRSCRATCH/trained_models/Lucie/instruction_assistant_only/mix_3 911 &
run_conversion $ALL_CCFRSCRATCH/trained_models/Lucie/instruction_assistant_only/mix_4 317 &
run_conversion $ALL_CCFRSCRATCH/trained_models/Lucie/instruction_assistant_only/mix_5 615 &

wait