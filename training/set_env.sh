# setup_env.sh

# Environement
module purge
module load cpuarch/amd 
module load anaconda-py3/2023.09 # Use this anaconda version already installed in Jean-Zay. If you're not on Jean-Zay, you have to install it.
module load cuda/12.1.0 # Use this cuda version already installed in Jean-Zay. If you're not on Jean-Zay, you have to install it.
module load gcc/12.2.0
conda activate lucie

# Variables
export OUTPUT_PATH=/gpfswork/rech/qgz/uzq54wg
export MEGATRON_DEEPSPEED_REPO=/linkhome/rech/gendjf01/uzq54wg/Lucie-Training/Megatron-DeepSpeed