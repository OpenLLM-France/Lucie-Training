#!/bin/bash

# H100
module purge
module load anaconda-py3/2024.06
module load cuda/12.1.0
module load gcc/12.2.0
conda activate lucie_eval

# A100
# module purge
# module load cpuarch/amd 
# module load anaconda-py3/2023.09 
# module load cuda/12.1.0
# module load gcc/12.2.0
# conda activate lucie_eval
