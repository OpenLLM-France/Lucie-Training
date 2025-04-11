# Scripts to train, evaluate and apply a tokenizer

This folder contains scripts to train, evaluate and apply a tokenizer.
The tokenizer is trained on the training data and then applied to the training, validation and test data.
The tokenizer is saved to disk and can be loaded and applied to new data.

## Train tokenizer

To train a tokenizer, run the following command:

```bash
python tokenizer_train.py [options] data1.parquet [data2.parquet ...]
```

## Tokenize data

Each tokenized dataset is represented by a set of `*.bin`, `*.idx` and `*.json` files.

The `*.bin` files contain the tokenized data, the `*.idx` files contain the indices of the beginning of each example in the `*.bin` files, and the `*.json` files contain the metadata of the dataset (number of tokens...).

The following subsections describe the 3 steps to generate those files.

### 1. Launch tokenization, parallelizing on subsets for each dataset

First, apply tokenizer on dataset(s) parallelizing on subsets of the data,
using the script [`tokenizer_apply.py`](tokenizer_apply.py).

This takes as argument a tokenizer and the name of the dataset(s) to tokenize, the name of an output folder,
and generates the tokenized data as a set of `*.bin` and `*.idx` files.

This can be done on Jean Zay by running [`sbatch slurm/apply_tokenizer.slurm`](slurm/tokenizer_apply.slurm) with a SLURM file like this
currently setup for RedPajama dataset (use `--datasets fine_web_edu` for FineWebEdu dataset for instance):
```slurm
#!/bin/bash
#SBATCH --job-name=tokens
#SBATCH --partition=cpu_p1
#SBATCH --account=qgz@cpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00

export RUN="python3 <<...>>/tokenization/tokenizer_apply.py \
    --tokenizer-name-or-path OpenLLM-France/Lucie-tokenizer-65k \
    --datasets red_pajama \
    --output <<...>>/lucie_tokens_65k \
    --workers 20 \
"

srun --output=tmp_slurm_output_tokens-$SLURM_JOBID.out --error=tmp_slurm_output_tokens-$SLURM_JOBID \
     --jobid $SLURM_JOBID bash -c "$RUN" 2>&1
```

### 2. Count number of tokens

Then, count the number of tokens in the dataset(s),
using the script [`count_tokens.py`](count_tokens.py).

This takes as argument an input folder with tokenized data and generate (missing) `*.json` files in it.

This can be done on Jean Zay by running [`sbatch slurm/count_tokens.slurm`](slurm/count_tokens.slurm) with a SLURM file like this:
```slurm
#!/bin/bash
#SBATCH --job-name=counttok
#SBATCH --partition=cpu_p1
#SBATCH --account=qgz@cpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00

export RUN="python3 <<...>>/tokenization/count_tokens.py  <<...>>/lucie_tokens_65k"

srun --output=tmp_slurm_output_count-$SLURM_JOBID.out --error=tmp_slurm_output_count-$SLURM_JOBID \
     --jobid $SLURM_JOBID bash -c "$RUN" 2>&1
```

### 3. Concatenate all tokenized files for each dataset

Finally, concatenate all tokenized files for subsets into a single file for each dataset,
using the script [`dataset_concat.py`](dataset_concat.py).

This takes as argument an input folder with tokenized data and generate (missing) `*.json` files.

This can be done on Jean Zay by running [`sbatch slurm/dataset_concat.slurm`](slurm/dataset_concat.slurm) with a SLURM file like this
currently setup to run only on RedPajama dataset (use `--only FineWebEdu` for FineWebEdu dataset for instance):
```slurm
#!/bin/bash
#SBATCH --job-name=token_concat
#SBATCH --partition=cpu_p1
#SBATCH --account=qgz@cpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00

export RUN="python3 <<path>>/tokenization/dataset_concat.py \
        <<...>>/lucie_tokens_65k \
        <<...>>/lucie_tokens_65k_grouped \
        --verbose \
        --only RedPajama \
"

srun --output=tmp_slurm_output_concat-$SLURM_JOBID.out --error=tmp_slurm_output_concat-$SLURM_JOBID.out \
     --jobid $SLURM_JOBID bash -c "$RUN" 2>&1
```

## Gather statistics in assets

### Count number of words

To count the number of words in the tokenized datasets, run the following command:

```bash
python data.py <<dataset_name>> --folder ../assets/stats_raw --ignore
```

### Compile all results

The script [`assets/compile_stats.py`](../assets/compile_stats.py) can be used to gather statistics on raw and tokenized datasets.
