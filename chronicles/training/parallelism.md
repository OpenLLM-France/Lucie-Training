
# Architecture: Llama

For now, we can use the architecture of Llama-2 6.7b. Megatron is not an official implementation of Llama, but it is still an abstract implementation of the GPT architecture. By specifying certain arguments, we can find the architectures of the most popular LLMs. For example, we can create a Llama architecture with this configuration:

- `HIDDEN_SIZE = 4096`
- `FFN_HIDDEN_SIZE = 11008`
- `NUM_LAYERS = 32`
- `NUM_HEADS = 32`
- `SEQ_LENGTH = 4096` (for Llama1, it is 2048 tokens).
- `NUM_KV_HEADS = 32`: We use the classic MultiHead Attention (MHA) mechanism. So `NUM_KV_HEADS = NUM_HEADS`. To use MultiQuery Attention (MQA) or Group Query Attention (GQA), this value needs to be changed.
- `ATTENTION_DROPOUT = 0`
- `HIDDEN_DROPOUT = 0`
- `USE_ROTARY_POSITION_EMBEDDING = true`
- `UNTIE_EMBEDDINGS_AND_OUTPUT_WEIGHTS = true`: We do not use weight tying, that is to say, the same matrix is used for input and output of the network (the one that predicts the next token).
- `SWIGLU`
- `NORMALIZATION = rmsnorm`
- `DISABLE_BIAS_LINEAR = true`

# Optimizer

For the optimizer, we use exactly the same hyperparameters given in the Llama-1 paper.

- `LR = 3e-4`
- `MIN_LR = 3e-5`
- `LR_WARMUP_STEPS = 2000`
- `WEIGHT_DECAY = 0.1`
- `GRAD_CLIP = 1`

