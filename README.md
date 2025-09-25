# Lucie Training

* [Setup](#setup)
  * [Clone the repository](#clone-the-repository)
  * [Environment setup](#environment-setup)
    * [With python virtual environment (conda)](#with-python-virtual-environment-conda)
    * [With Docker](#with-docker)
  * [Install Megatron-Deepspeed](#install-megatron-deepspeed)
* [Train a model](#train-a-model)
  * [1. Pretraining (first main phase)](#1-pretraining-first-main-phase)
  * [2. Context Extension](#2-context-extension)
  * [3. Annealing](#3-annealing)
  * [4. Instruct-Tuning and Finetuning](#4-instruct-tuning-and-finetuning)
* [Model conversion](#model-conversion)
  * [From Megratron-Deepspeed to transformers](#from-megratron-deepspeed-to-transformers)
  * [From LORA (PEFT) to full weights](#from-lora-peft-to-full-weights)
  * [Quantize models](#quantize-models)
* [Acknowledgment/Support](#acknowledgmentsupport)

## Setup

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

## Train a model

### 1. Pretraining (first main phase)

TODO

### 2. Context Extension

TODO

### 3. Annealing

TODO

### 4. Instruct-Tuning and Finetuning

TODO

## Model conversion

### From Megratron-Deepspeed to transformers

Converting a Megatron-Deepspeed model/checkpoint in transformers format
can be done by running the following in the folder with the clone of [OpenLLM-France's fork of Megatron-DeepSpeed](https://github.com/OpenLLM-France/Megatron-DeepSpeed)
```bash
MEGATRON_CHECKPOINT=... # input path
UNIVERSAL_CHECKPOINT=... # output path (1st step)
TRANSFORMERS_CHECKPOINT=... # output path (final)

if [ ! -d $UNIVERSAL_CHECKPOINT ]; then
    # DS to Universal Megatron (merge chunks of models that were spread overs GPUs)
    python tools/convert_checkpoint/ds_to_universal.py --input_folder $MEGATRON_CHECKPOINT --output_folder $UNIVERSAL_CHECKPOINT
fi

if [ ! -d $TRANSFORMERS_CHECKPOINT ]; then
    # Universal Megatron to transformers
    python tools/convert_checkpoint/universal_to_hf_llama.py --input_folder $UNIVERSAL_CHECKPOINT --output_folder $TRANSFORMERS_CHECKPOINT --max_shouiard_size 5GB
fi
```

### From LORA (PEFT) to full weights

If you have LORA weights from Parameter Efficient FineTuning (PEFT),
you can combine those weights with the base model to have the "full" model.

This can typically be done with this kind of script on CPU:
```python
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("lora", type=str, default=None)
parser.add_argument("base", type=str, default=None)
parser.add_argument("output", type=str, default=None)
parser.add_argument("device", type=str, default="cpu")
args = parser.parse_args()

path_base = args.base
path_instruct = args.lora
path_output = args.output
device = torch.device(args.device)

tokenizer = AutoTokenizer.from_pretrained(path_base)

model = AutoModelForCausalLM.from_pretrained(
    path_base,
    torch_dtype=torch.bfloat16
)

model = PeftModel.from_pretrained(model, path_instruct)
model.to(device).eval()

model = model.merge_and_unload()
model.eval()

model.save_pretrained(path_output)
```

### Quantize models

Quantification can be done from models in `transformers` format.

Install [llama.cpp](https://github.com/ggerganov/llama.cpp):
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build --config Release
python3 -m pip install -r requirements.txt
```

Run the conversion of the model packaged for Hugging Face,
choosing the desired quantization method:
```bash
# Convert to GGUF FP16 format
python3 convert_hf_to_gguf.py /path/to/model/transformers/ --outfile custom-name-f16.gguf --outtype f16

# Quantize model weights
./llama-quantize custom-name-f16.gguf custom-name-q4_k_m.gguf q4_k_m
```

## Acknowledgment/Support

This repository is maintained by [OpenLLM-France](https://github.com/OpenLLM-France), a group of researchers and engineers working on large language models:
* [Jérôme Louradour](https://github.com/Jeronymous) -- support him on [Buy Me A Coffee](https://buymeacoffee.com/jeronymous)
* [Olivier Gouvert](https://github.com/Oligou)
* [Julie Hunter](https://github.com/jhunter19)
* [Yaya Sy](https://github.com/yaya-sy)
* [Christophe Cerisara](https://github.com/cerisara)
