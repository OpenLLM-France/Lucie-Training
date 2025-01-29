#!/bin/bash
#SBATCH --job-name=counttok
#SBATCH --partition=cpu_p1
#SBATCH --account=qgz@cpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00

DATE=`date +"%Y-%m-%d--%H-%M-%S"`

module purge
module load anaconda-py3/2024.06
module load cuda/12.1.0
module load gcc/12.2.0
conda activate /lustre/fswork/projects/rech/fwx/commun/Lucie_h100

export RUN="python3 ~/Lucie-Training/tokenization/count_tokens.py  \
        /lustre/fsn1/projects/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_65k_instruction_4kpad \
        "
        
srun --output=tmp_slurm_output_count-$DATE.out --error=tmp_slurm_output_count-$DATE.out \
     --jobid $SLURM_JOBID bash -c "$RUN" 2>&1
