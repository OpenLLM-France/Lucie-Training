#!/bin/bash

module purge
module load cpuarch/amd 
module load anaconda-py3/2023.09 
conda activate /gpfsscratch/rech/qgz/uzq54wg/datatrove 

python domains.py --dump-to-process $1 --language $2 --main-output-path $SCRATCH/processed_redpajama/v2