# Lucie Training

## 1. Setup

### Create the conda environment
```bash
module load anaconda-py3/2023.09
conda create -n lucie python=3.10
```

### Set the conda environment
```bash
conda activate lucie
module load cuda/12.1.0 # use the cuda already installed in Jean-Zay
```

### Install torch and ninja
We recommend to use the latest stable torch from https://pytorch.org/get-started/locally/
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install ninja # Will be needed for compilation
```

### Install apex
```bash
git clone https://github.com/NVIDIA/apex
cd apex/
pip install -r requirements.txt
module load cuda/12.1.0
MAX_JOBS=4 pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./ 
# You can run this on CPUs instance as it's compute intensive. You may encounter some errors here, just rerun this command. If it still don't work, consider lowering the value of MAX_JOBS
cd ...
```
### Install Megatron-Deepspeed
```bash
git clone https://github.com/OpenLLM-France/Megatron-DeepSpeed.git
cd  Megatron-DeepSpeed/
pip install -e .
pip install deepspeed==0.12.6
cd ...
```

### Install any remaining python dependencies 
```bash
pip install six
pip install transformers
# maybe flash_attention (or others) in the future
```

### Activate environment on Jean-Zay
```bash
module load anaconda-py3/2023.09
conda activate lucie
module load cuda/12.1.0
```

### Run training for debug

```bash
sh scripts/training/pretrain_llama.sh <MEGATRON_REPO> <CACHE_FOLDER> <CHECKPOINTS_FOLDER>
```

