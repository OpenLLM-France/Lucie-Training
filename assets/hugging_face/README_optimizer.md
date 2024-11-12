# Model Card

This repository contains universal checkpoints in DeepSpeed format for the [Lucie-7B model](https://huggingface.co/OpenLLM-France/Lucie-7B),
which was trained using [this repository of code](https://github.com/OpenLLM-France/Lucie-Training)
based on [a fork of `Megatron-Deepspeed`](https://github.com/OpenLLM-France/Megatron-DeepSpeed).

Each checkpoint is in a subbranch (revision), which names specifies the number of training steps.
For instance `step0400000` corresponds to the checkpoint after 4M training steps.

Available revisions include:
* ["`step0005000`"](https://huggingface.co/OpenLLM-France/Lucie-7B-optimizer-states/tree/step0005000), ["`step0010000`"](https://huggingface.co/OpenLLM-France/Lucie-7B-optimizer-states/tree/step0010000), ["`step0015000`"](https://huggingface.co/OpenLLM-France/Lucie-7B-optimizer-states/tree/step0015000), ["`step0020000`"](https://huggingface.co/OpenLLM-France/Lucie-7B-optimizer-states/tree/step0020000): each 5000 steps for the first pre-training steps.
* ["`step0025000`"](https://huggingface.co/OpenLLM-France/Lucie-7B-optimizer-states/tree/step0025000), ["`step0050000`"](https://huggingface.co/OpenLLM-France/Lucie-7B-optimizer-states/tree/step0050000), ["`step0075000`"](https://huggingface.co/OpenLLM-France/Lucie-7B-optimizer-states/tree/step0075000), ["`step0100000`"](https://huggingface.co/OpenLLM-France/Lucie-7B-optimizer-states/tree/step0100000), ..., ["`step0750000`"](https://huggingface.co/OpenLLM-France/Lucie-7B-optimizer-states/tree/step0750000): each 25000 steps from 25k to 750k steps.
* ["`step0753851`"](https://huggingface.co/OpenLLM-France/Lucie-7B-optimizer-states/tree/step0753851): last pre-training step before context extension and annealing.

Those checkpoints are provided so that the model can be retrained from a given point.

## Contact

contact@openllm-france.fr
