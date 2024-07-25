# Scripts to train, evaluate and apply a tokenizer

This folder contains scripts to train, evaluate and apply a tokenizer.
The tokenizer is trained on the training data and then applied to the training, validation and test data.
The tokenizer is saved to disk and can be loaded and applied to new data.

## Train tokenizer

To train a tokenizer, run the following command:

```bash
python tokenizer_train.py [options]
```

## Tokenize data

### 1. Launch tokenization

First, apply tokenizer on dataset(s) parallelizing on subsets of the data.
This is done with the script `tokenizer_apply.py`.
This can be done on Jean Zay running the SLURM script [`sbatch slurm/apply_tokenizer.slurm`](slurm/tokenizer_apply.slurm)
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

export RUN="python3 <<path>>/tokenization/tokenizer_apply.py \
        --datasets red_pajama \
        --output /gpfsssd/scratch/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_65k \
        --workers 20 \
        "

srun --output=tmp_slurm_output_tokens-$SLURM_JOBID.out --error=tmp_slurm_output_tokens-$SLURM_JOBID \
     --jobid $SLURM_JOBID bash -c "$RUN" 2>&1
```

Clean unfinished jobs (and restart as long as necessary):

```bash
bash clean_unterminated_tokens.sh | xargs rm
```

### 2. Count number of tokens

Then, count the number of tokens in the dataset(s).
This is done with the script `count_tokens.py`.
This can be done on Jean Zay running the SLURM script [`sbatch slurm/count_tokens.slurm`](slurm/count_tokens.slurm):
```slurm
#!/bin/bash
#SBATCH --job-name=count_tokens
#SBATCH --partition=cpu_p1
#SBATCH --account=qgz@cpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00

export RUN="python3 <<...>>/tokenization/count_tokens.py  \
        /gpfsssd/scratch/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_65k \
        "

srun --output=tmp_slurm_output_count-$SLURM_JOBID.out --error=tmp_slurm_output_count-$SLURM_JOBID \
     --jobid $SLURM_JOBID bash -c "$RUN" 2>&1
```

### 3. Concatenate all tokenized files for each dataset

Finally, concatenate all tokenized files into a single file.
This is done with the script `dataset_concat.py`.
This can be done on Jean Zay running the SLURM script [`sbatch slurm/dataset_concat.slurm`](slurm/dataset_concat.slurm)
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
        /gpfsssd/scratch/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_65k \
        /gpfsssd/scratch/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_65k_grouped \
        --verbose --only RedPajama"

srun --output=tmp_slurm_output_concat-$SLURM_JOBID.out --error=tmp_slurm_output_concat-$SLURM_JOBID.out \
     --jobid $SLURM_JOBID bash -c "$RUN" 2>&1
```
