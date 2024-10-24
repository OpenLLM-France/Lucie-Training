#!/bin/bash

module purge
module load anaconda-py3/2023.09 
conda activate /lustre/fsn1/projects/rech/qgz/uzq54wg/envs/datatrove

dataset=$HOME/Lucie-Training/assets/RedPajama-Data-V2 

python base.py --dataset-name $dataset --dump-to-process $1 --language $2 --main-output-path $SCRATCH/processed_redpajama