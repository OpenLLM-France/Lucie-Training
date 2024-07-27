#!/bin/bash
#SBATCH --job-name=counttok
#SBATCH --partition=cpu_p1
#SBATCH --account=qgz@cpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00

DATE=`date +"%Y-%m-%d--%H-%M-%S"`

export RUN="python3 /gpfs7kw/linkhome/rech/gendjf01/uov59an/src/Bloom-NG-Training/tokenization/count_tokens.py  \
        /gpfsssd/scratch/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_65k \
        "

srun --output=tmp_slurm_output_count-$DATE.out --error=tmp_slurm_output_count-$DATE.out \
     --jobid $SLURM_JOBID bash -c "$RUN" 2>&1
