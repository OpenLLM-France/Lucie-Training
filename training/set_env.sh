#!/bin/bash

# Environement
module purge
module load anaconda-py3/2024.06
module load cuda/12.1.0
module load gcc/12.2.0
conda activate /lustre/fswork/projects/rech/fwx/commun/Lucie_h100

# Variables
export CHECKPOINT_PATH=$ALL_CCFRSCRATCH/checkpoints/
export TENSORBOARD_PATH=$ALL_CCFRSCRATCH/tensorboard/
export LOGS_PATH=$ALL_CCFRSCRATCH/lucie-logs/

export MEGATRON_DEEPSPEED_REPO=/linkhome/rech/gendjf01/uzq54wg/Lucie-Training/Megatron-DeepSpeed