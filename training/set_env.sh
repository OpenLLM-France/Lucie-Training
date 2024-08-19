#!/bin/bash

# Environement
module purge
module load cpuarch/amd 
module load anaconda-py3/2023.09 # Use this anaconda version already installed in Jean-Zay. If you're not on Jean-Zay, you have to install it.
module load cuda/12.1.0 # Use this cuda version already installed in Jean-Zay. If you're not on Jean-Zay, you have to install it.
module load gcc/12.2.0
conda activate lucie

# Variables
export OUTPUT_PATH=/gpfsscratch/rech/qgz/ull45hr
export MEGATRON_DEEPSPEED_REPO=/linkhome/rech/gendjf01/ull45hr/Lucie-Training/Megatron-DeepSpeed
