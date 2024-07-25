#!/bin/bash
#SBATCH --job-name=token_concat
#SBATCH --partition=cpu_p1
#SBATCH --account=qgz@cpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00

export RUN="python3 /gpfs7kw/linkhome/rech/gendjf01/uov59an/src/Bloom-NG-Training/tokenization/dataset_concat.py \
        /gpfsssd/scratch/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_65k \
        /gpfsssd/scratch/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_65k_grouped \
        --verbose --only RedPajama"

srun --output=tmp_slurm_output_concat-$SLURM_JOBID.out --error=tmp_slurm_output_concat-$SLURM_JOBID.out \
     --jobid $SLURM_JOBID bash -c "$RUN" 2>&1
