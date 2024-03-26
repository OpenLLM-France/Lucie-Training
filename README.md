# Lucie Training

## 1. Setup

### Create the conda environment
```bash
module load anaconda-py3/2023.09
conda create -n lucie-cuda211 python=3.10
```

### Set the conda environment
```bash
conda activate lucie-cuda211
module load cuda/12.1.0 # use the cuda already installed in Jean-Zay
module load ninja/1.10.0 # used for fast compilation
```

### Install the latest stable torch from https://pytorch.org/get-started/locally/
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Install apex
```bash
git clone https://github.com/NVIDIA/apex
cd apex/
pip install -r requirements.txt
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

### Utilisation
```bash
module load anaconda-py3/2023.09
conda activate lucie-cuda211
module load cuda/12.1.0
```

## 2. Training Dataset Storage and Organization

Training dataset is stored in [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) files. This format is designed to store and manage large amounts of data efficiently.

The composition of the dataset includes a variety of document types. These document types should be organized based on their likelihood of occurrence during the training process. For example, empirical studies have shown that incorporating a mix of 10% to 15% code-related documents into the training set can significantly enhance the model's ability to perform logical reasoning and understand code syntax. In that case, your dataset directory could be composed of two directories as presented below.

```
/path/to/dataset
├── english
│   ├── english_00001-of-10000.hdf5
│   ├── ...
│   └── english_10000-of-10000.hdf5
└── code
    ├── the-stack_00001-of-99920.hdf5
    ├── ...
    └── the-stack_99920-of-99920.hdf5
```

The likelihood of occurrence for different document types should be cataloged in a JSON file, as illustrated in the following example. Here, documents pertaining to code are allocated a likelihood value of 0.13, suggesting that documents associated with code are anticipated to constitute roughly 13% of the entire dataset.

```
{
    "english": {
        "directory": "/path/to/dataset/english",
        "probability": 0.87
    },
    "code": {
        "directory": "/path/to/dataset/code",
        "probability": 0.13
    }
}
```