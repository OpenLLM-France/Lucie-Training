#!/bin/bash
#SBATCH --job-name=RP_tokenize
#SBATCH --account=qgz@cpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --partition=prepost
#SBATCH --time=20:00:00

DATE=`date +"%Y-%m-%d--%H-%M-%S"`

module purge
module load anaconda-py3/2024.06
module load cuda/12.1.0
module load gcc/12.2.0
conda activate /lustre/fswork/projects/rech/fwx/commun/Lucie_h100

# export RUN="python3 /gpfs7kw/linkhome/rech/gendjf01/uov59an/src/Bloom-NG-Training/tokenization/tokenizer_apply.py  \
#         --datasets red_pajama \
#         --output /gpfsssd/scratch/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_65k \
#         --workers 20 \
#         "

export RUN="python3 ~/Lucie-Training/tokenization/tokenizer_apply.py  \
        --datasets $1 \
        --output /lustre/fsn1/projects/rech/fwx/commun/preprocessed_data/Lucie/lucie_tokens_65k_annealing \
        --workers 20 \
        "

srun --output=tmp_slurm_output_tokens-RP-$DATE.out --error=tmp_slurm_output_tokens-RP-$DATE.out \
     --jobid $SLURM_JOBID bash -c "$RUN" 2>&1
