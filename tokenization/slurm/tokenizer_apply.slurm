#!/bin/bash
#SBATCH --job-name=RP_tokenize
#SBATCH --partition=cpu_p1
#SBATCH --account=qgz@cpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00

export RUN="python3 /gpfs7kw/linkhome/rech/gendjf01/uov59an/src/Bloom-NG-Training/tokenization/tokenizer_apply.py  \
        --datasets red_pajama \
        --output /gpfsssd/scratch/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_65k \
        --workers 20 \
        "

srun --output=tmp_slurm_output_tokens-$SLURM_JOBID.out --error=tmp_slurm_output_tokens-$SLURM_JOBID \
     --jobid $SLURM_JOBID bash -c "$RUN" 2>&1
