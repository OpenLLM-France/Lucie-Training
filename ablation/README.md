# Ablation study

## Data Proportions study

The idea here is to challenge the model with different proportions of data (data mixtures). Considering previous work one could hope that experiments on small scale model (e.g. 80M parameters) should be a good proxy to estimate a larger scale model (e.g. 7B parameters) optimal data mixture.

### Validate the hypothesis

The idea is to validate the hypothesis that a small scale model can be used to estimate the optimal data mixture for a larger scale model. Here, we propose to train a small scale model (80M parameters) on different data mixtures and then train a larger scale model (410M parameters) on the same data mixtures.

Then, the idea is to compare the perplexity of both models on a test set and see if the rank of the different domains is preserved between the two models when trained on the same data mixture.

For this first experiment, we will train both models on 30B tokens.

#### Prerequisites

For those experiments, one will need to change the Megatron-DeepSpeed branch.

Supposing that you are in the `Lucie-Training` directory, you can do the following:

```bash
cd Megatron-DeepSpeed
git checkout perplexity
cd ..
```

#### Train the models with different mixtures

To train the models, we will use the following command:

```bash
sbatch --nodes=2 --time=04:30:00 --job-name=ablation-datamix-lucie80m-config00-30ksteps ablation/ablation_training.slurm --model_config lucie80m --data_config config00
```

To determine the different options you can follow the following table that will help you to determine the different options depending on the model's size:

| Model size | model_config | nodes | time |
|------------|--------------|-------------|------|
| 80M        | lucie80m     | 2           | 04:30:00 |
| 410M       | lucie410m    | 4           | 16:00:00 |

To set the `data_config` parameter the following options are available:

| data_config | config_path |
|-------------|--------------|
| config00    | data_config/config00.yaml |
| config01    | data_config/config01.yaml |

Finally, when all the above are decided you should set the `--job-name` parameter to: `ablation-datamix-<model_config>-<data_config>-30ksteps`. It will be helpful later to compute the different perplexities.

#### Compute the perplexities

To compute the perplexities, we will use the following command:

```bash
sbatch --nodes=2 --time=01:30:00 --job-name=perplexity-ablation-datamix-lucie80m-config00-30ksteps ablation/ablation_perplexity.slurm --model_config lucie80m --data_config config00
```

The same options as for the training should be used as this scripts will search the checkpoints in the same directory as the training scripts.

Finally, the `--job-name` parameter should be set to: `perplexity-ablation-datamix-<model_config>-<data_config>-30ksteps`.

#### Analyze the results
**WIP**

### Find the optimal data mixture for Lucie80M

The idea here is to find the optimal data mixture for the Lucie80M model. To do so, we will train the model on different data mixtures and then evaluate the perplexity on a test set. We will then compare the different perplexities and determine the optimal data mixture.

#### The data mixtures
**WIP**

#### Train the models with different mixtures
**WIP**

#### Compute the perplexities
**WIP**

#### Analyze the results
**WIP**
