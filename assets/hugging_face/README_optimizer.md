# Model Card

This repository contains universal checkpoints in DeepSpeed format for the [Lucie-7B model](https://huggingface.co/OpenLLM-France/Lucie-7B),
which was trained using [this repository of code](https://github.com/OpenLLM-France/Lucie-Training)
based on [a fork of `Megatron-Deepspeed`](https://github.com/OpenLLM-France/Megatron-DeepSpeed).

Each checkpoint is in a subbranch (revision), which names specifies the number of training steps.
For instance `step0400000` corresponds to the checkpoint after 4M training steps.

Those checkpoints are provided so that the model can be retrained from a given point.

## Contact

contact@openllm-france.fr
