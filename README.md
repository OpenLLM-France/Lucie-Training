# Lucie Training

## 1. Requirements

**Step 1** - Create a Python (>=3.9) environment and install PyTorch according to your hardware configuration.

```shell
$ conda create -n lucie-training python=3.9
$ conda activate lucie-training
$ # Install PyTorch
```

**Step 2** - Install the library.

```shell
$ pip install -e .
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