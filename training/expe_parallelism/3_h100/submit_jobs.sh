#!/bin/bash

# Function to submit the job with given parameters
submit_job() {
    local nodes=$1
    local TP=$2
    local PP=$3
    local GLOBAL_BATCH_SIZE=$4
    local MICRO_BATCH_SIZE=$5
    local ZERO_STAGE=$6

    local job_name=lucie7b-${nodes}nodes-tp${TP}-pp${PP}-z${ZERO_STAGE}-gbz${GLOBAL_BATCH_SIZE}-mbz${MICRO_BATCH_SIZE}

    sbatch --export=ALL,TP=$TP,PP=$PP,GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE,MICRO_BATCH_SIZE=$MICRO_BATCH_SIZE,ZERO_STAGE=$ZERO_STAGE --nodes $nodes --qos=qos_gpu-t3 --job-name=$job_name lucie7b.slurm
}

# Best config
submit_job 32 2 4 1024 4 0