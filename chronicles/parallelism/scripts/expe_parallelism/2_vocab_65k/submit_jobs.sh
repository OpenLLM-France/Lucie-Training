#!/bin/bash

# Function to submit the job with given parameters
submit_job() {
    local nodes=$1
    local TP=$2
    local PP=$3
    local GLOBAL_BATCH_SIZE=$4
    local MICRO_BATCH_SIZE=$5
    local ZERO_STAGE=$6
    local qos

    local job_name=lucie7b-v65k-${nodes}nodes-tp${TP}-pp${PP}-z${ZERO_STAGE}-gbz${GLOBAL_BATCH_SIZE}-mbz${MICRO_BATCH_SIZE}
    
    if [ $nodes -gt 4 ]; then
        qos=qos_gpu-t3
    else
        qos=qos_gpu-dev
    fi

    sbatch --export=ALL,TP=$TP,PP=$PP,GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE,MICRO_BATCH_SIZE=$MICRO_BATCH_SIZE,ZERO_STAGE=$ZERO_STAGE --nodes $nodes --qos=$qos --job-name=$job_name lucie7b-v65k.slurm
}

# Best config with v=65k
# submit_job 16 2 2 256 2 0
# submit_job 16 2 2 512 2 0
# submit_job 16 2 2 1024 2 0
# submit_job 32 2 2 1024 2 0
# submit_job 4 2 2 1024 2 0
submit_job 32 2 4 1024 2 0
submit_job 32 2 4 1024 4 0