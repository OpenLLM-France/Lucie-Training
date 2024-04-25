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
module load anaconda-py3/2023.09 # Use this anaconda version already installed in Jean-Zay. If you're not on Jean-Zay, you have to install it.
module load cuda/12.1.0 # Use this cuda version already installed in Jean-Zay. If you're not on Jean-Zay, you have to install it.
module load cpuarch/amd # Specif to Jean-Zay only. Ignore this if you're not on Jean-Zay
conda activate lucie
```

> Tips for Jean-Zay: you can add these lines to your `$HOME/.bash_profile` file, if you always want to load these modules when you connect.

Install torch:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
> We recommend to use the latest stable torch from https://pytorch.org/get-started/locally/

In the `Lucie-Training` folder, install the [python dependencies](requirements.txt):
```bash
pip install -r requirements.txt
```

Install ninja:
```bash
pip install ninja
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



<!-- ### Run training for debug

```bash
sh scripts/training/pretrain_llama.sh <MEGATRON_REPO> <CACHE_FOLDER> <CHECKPOINTS_FOLDER>
```
 -->
