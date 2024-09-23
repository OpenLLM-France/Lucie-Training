#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=4                   
#SBATCH --gpus-per-task=1            
#SBATCH --cpus-per-task=24           
#SBATCH --hint=nomultithread         
#SBATCH --constraint=h100            
#SBATCH --time 2:00:00               
#SBATCH --output=slurm_logs/%x-%j.out          
#SBATCH --job-name=lucie_conversion

source set_env.sh
cd /lustre/fshomisc/home/rech/gendjf01/uov59an/src/Bloom-NG-Training/Megatron-DeepSpeed/
export PYTHONPATH=/lustre/fshomisc/home/rech/gendjf01/uov59an/src/Bloom-NG-Training/Megatron-DeepSpeed

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

    if [ ! -d $TRANSFORMERS_CHECKPOINT_PATH ]; then
        # Universal to Transformer
        srun --exclusive -n 1 python tools/convert_checkpoint/universal_to_hf_llama.py --input_folder $UNIVERSAL_CHECKPOINT_PATH --output_folder $TRANSFORMERS_CHECKPOINT_PATH 
    fi
}

### LUCIE PRETRAINED
i=25000
while [ $i -le 350000 ]; do
    run_conversion $ALL_CCFRSCRATCH/trained_models/Lucie/pretrained $i &
    i=$((i + 25000))
done

### CRISIS HP
run_conversion $ALL_CCFRSCRATCH/trained_models/Lucie/ablation_pythia_hp 5000 &
run_conversion $ALL_CCFRSCRATCH/trained_models/Lucie/ablation_pythia_hp 10000 &
run_conversion $ALL_CCFRSCRATCH/trained_models/Lucie/ablation_pythia_hp 14204 &

### CRISIS PILE
run_conversion $ALL_CCFRSCRATCH/trained_models/Lucie/pretrained_on_pile 5000 &

wait