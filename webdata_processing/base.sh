#!/bin/bash

dataset=$HOME/Lucie-Training/assets/RedPajama-Data-V2 

python base.py --dataset-name $dataset --dump-to-process $1 --language $2 --main-output-path $SCRATCH/processed_redpajama/v3