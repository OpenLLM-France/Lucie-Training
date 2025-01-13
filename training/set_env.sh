#!/bin/bash

# Jean-Zay Modules
module purge
module load arch/h100
module load anaconda-py3/2024.06
module load cuda/12.1.0
module load gcc/12.2.0

# Python Environment
conda activate /lustre/fswork/projects/rech/fwx/commun/Lucie_h100

# Environment Variables
# - source
export SRC_DIR="/linkhome/rech/xxxxxx00/xxx00xx/Lucie-Training"
export MEGATRON_DEEPSPEED_REPO="$SRC_DIR/Megatron-DeepSpeed"
export ASSET_DIR="$SRC_DIR/assets"
# - (tokenized) data
export TOKENS_DATA_DIR="$ALL_CCFRSCRATCH/preprocessed_data/Lucie/lucie_tokens_65k_grouped"
# - output model (and training logs)
export OUTPUT_DIR="$ALL_CCFRSCRATCH/trained_models/Lucie"