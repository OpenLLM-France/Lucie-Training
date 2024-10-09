# Lucie Training

* [1. Setup](#1-setup)
  * [Clone the repository](#clone-the-repository)
  * [Environment setup](#environment-setup)
    * [With python virtual environment (conda)](#with-python-virtual-environment-conda)
    * [With Docker](#with-docker)
  * [Install Megatron-Deepspeed](#install-megatron-deepspeed)

## 1. Setup

### Clone the repository

Clone this repository:
```bash
git clone git@github.com:OpenLLM-France/Lucie-Training.git
cd Lucie-Training/
```

Inside the `Lucie-Training` folder, clone [our fork of Megatron-Deepspeed](https://github.com/OpenLLM-France/Megatron-DeepSpeed):
```bash
git clone https://github.com/OpenLLM-France/Megatron-DeepSpeed.git
```

### Environment setup

#### With python virtual environment (conda)

This is the recommended way to install the dependencies on supercomputers like Jean-Zay.

Create a conda environment (first time only):
```bash
module load anaconda-py3/2023.09 # Use this anaconda version already installed in Jean-Zay. If you're not on Jean-Zay, you have to install it.
conda create -n lucie python=3.10
```

Set the conda environment:
```bash
module load cpuarch/amd # Specif to Jean-Zay only. Ignore this if you're not on Jean-Zay
module load anaconda-py3/2023.09 # Use this anaconda version already installed in Jean-Zay. If you're not on Jean-Zay, you have to install it.
module load cuda/12.1.0 # Use this cuda version already installed in Jean-Zay. If you're not on Jean-Zay, you have to install it.
module load gcc/12.2.0
conda activate lucie
```

> Tips for Jean-Zay: you can add these lines to your `$HOME/.bash_profile` file, if you always want to load these modules when you connect.

Install torch:
```bash
conda install pytorch=2.3.0 torchvision=0.18.0 torchaudio=2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```
> We recommend to use the latest stable torch from https://pytorch.org/get-started/locally/

In the `Lucie-Training` folder, install the [python dependencies](requirements.txt):
```bash
pip install -r requirements.txt
```

Install ninja:
```bash
conda install ninja
```
> `ninja` will be needed for compiling some parts of Megatron-Deepspeed

Install apex:
```bash
git clone https://github.com/NVIDIA/apex
cd apex/
pip install -r requirements.txt
MAX_JOBS=4 pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./ 
cd ..
```
> This compilation is compute intensive. You may encounter some errors here, just rerun this command. If it still don't work, consider lowering the value of MAX_JOBS


#### With Docker

In the `Lucie-Training` folder, build the docker image:
```bash
docker build . -f Dockerfile -t lucie:latest
```

Run the docker container, mounting the necessary folders:
```bash
docker run -it --rm --gpus all \
    -v $HOME:$HOME \
    -v .../Lucie-Training/Megatron-DeepSpeed:/workspace/megatron \
    --workdir .../Lucie-Training \
    --name lucie_workspace
    lucie:latest
```

> In case of cuda issues, you may have to mount `-v /usr/local/lib/python3.10/dist-packages/nvidia/cudnn:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn`

Connect to the container in another terminal:
```bash
docker exec -it lucie_workspace bash
```

### Install Megatron-Deepspeed

At this step, we assume that the following things have been installed: torch, ninja and apexInstall ninja:

Install Megatron-Deepspeed, inside the `Lucie-Training` folder:
```bash
cd  .../Lucie-Training/Megatron-DeepSpeed/
pip install -e .
cd ..
```

Notes:
* When running training with Megatron-Deepspeed for the first time on a given architecture,
  some "fused kernels" will be built by `ninja` under the path `Megatron-DeepSpeed/megatron/fused_kernels/build`.
  Moving the `Megatron-DeepSpeed` folder will require to rebuild at the next call.
* It is important to have `deepspeed==0.12.6` installed (more recent versions will not work with this fork of `Megatron-DeepSpeed`).
  See [requirements.txt](requirements.txt).


#### Check insallation

Some useful checks that can be made (based on the classical errors we saw when using our fork of Megatron-Deepspeed):
```bash
python -c "from torch.distributed.elastic.agent.server.api import log" # Check if torch is compatible with Megatron-Deepspeed
python -c "from flash_attn import flash_attn_qkvpacked_func, flash_attn_func" # Check if flash_attn is correctly installed
python -c "import amp_C" # Check if apex is correctly installed
```

WIP (trying to have minimal command to check the pre-training)
```bash
srun --ntasks=1 --gres=gpu:1 -C a100 -A qgz@a100 --qos=qos_gpu-dev --time=00:03:00 python -c "import os, subprocess; os.environ['MASTER_ADDR']= subprocess.run('scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1', shell=True, capture_output=True, text=True).stdout; os.environ['MASTER_PORT'] = '6000'; from megatron.initialize import initialize_megatron; initialize_megatron(); print('ok')" --micro-batch-size 2 --num-layers 2 --use-dataset-only True --seq-length 100 --tokenizer-type PretrainedFromHF --tokenizer-name-or-path OpenLLM-France/Lucie-tokenizer-65k --no-pipeline-parallel
```

<!-- ### Run training for debug

```bash
sh scripts/training/pretrain_llama.sh <MEGATRON_REPO> <CACHE_FOLDER> <CHECKPOINTS_FOLDER>
```
 -->
