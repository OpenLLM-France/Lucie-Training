"""
This is a script to calculate the perplexity of a language model on a given validation set.
"""
import os

import math
import json
import csv

import numpy as np
import torch
import torch.distributed as dist

from megatron import get_args
from megatron.initialize import initialize_megatron
from megatron import print_rank_0, is_last_rank
from megatron import get_tokenizer
from megatron.data.gpt_dataset import GPTDataset, get_indexed_dataset_
from megatron.core import mpu
from megatron.core import tensor_parallel
from megatron.arguments import core_transformer_config_from_args
from megatron.model import GPTModel, GPTModelPipe
from megatron.utils import get_ltor_masks_and_position_ids
from deepspeed.runtime.utils import see_memory_usage
from megatron.model.rotary_pos_embedding import RotaryEmbedding

from megatron.training import setup_model_and_optimizer
from megatron.core.enums import ModelType

# from megatron.data.data_samplers import build_pretraining_data_loader
from megatron.data.data_samplers import MegatronPretrainingSampler

import deepspeed
from deepspeed.accelerator.real_accelerator import get_accelerator

## Code from pretrain_gpt.py
def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building GPT model ...')
    see_memory_usage(f"Before Building Model", force=True)

    args = get_args()
    config = core_transformer_config_from_args(args)
    with deepspeed.zero.Init(sequence_data_parallel_group=mpu.get_sequence_data_parallel_group(),
                             remote_device=None if args.remote_device == 'none' else args.remote_device,
                             config_dict_or_path=args.deepspeed_config,
                             enabled=args.zero_stage == 3,
                             mpu=mpu):
        if args.deepspeed and not args.no_pipeline_parallel:
            model = GPTModelPipe(
                config=config,
                num_tokentypes=0,
                parallel_output=True
            )
            # This is a hack to give us a reference to get_batch_pipe from within training.py
            # We need to call model.set_batch_fn after deepspeed.initialize
            model._megatron_batch_fn = get_batch_pipe

            # Predompute the attention mask and store it in args. This avoids having to
            # pipeline it as an activation during training. The mask is constant, and thus
            # we can reuse it.
            attention_mask = torch.tril(torch.ones(
                (1, args.seq_length, args.seq_length), device=get_accelerator().current_device_name())).view(
                    1, 1, args.seq_length, args.seq_length)

            # Convert attention mask to binary:
            attention_mask = (attention_mask < 0.5)
            if args.fp16:
                attention_mask = attention_mask.half()
            elif args.bf16:
                attention_mask = attention_mask.bfloat16()

            # Attention mask must be bool.
            args.attn_mask = attention_mask.to(torch.bool)

            # For prertaining, since sequence length is fixed, cache rotary embedding in args, to avoid communicating around
            if args.use_rotary_position_embeddings:
                rotary_dim = args.hidden_size // args.num_attention_heads \
                    if args.kv_channels is None else args.kv_channels

                if args.rotary_percent < 1.0:
                    rotary_dim = int(rotary_dim * args.rotary_percent)

                # partial rotary embeddings, which is better than full rotary
                # Wang and Komatsuzaki et al
                # https://github.com/kingoflolz/mesh-transformer-jax/
                rotary_pos_emb = RotaryEmbedding(rotary_dim)(args.seq_length).to(
                    get_accelerator().current_device_name())
                args.rotary_pos_emb = rotary_pos_emb

        else:
            model = GPTModel(
                config=config,
                num_tokentypes=0,
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process
            )
    see_memory_usage(f"After Building Model", force=True)
    return model

def get_batch_pipe(data):
    """Modification of `get_batch` to work on `next(data_iterator)` instead of `data_iterator`"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)
    if args.curriculum_learning_legacy and args.curriculum_seqlen < tokens.size()[1]:
        # seqlen-based curriculum learning
        # tokens, position_ids, labels, loss_mask have size [batch size, seqlen]
        tokens = tokens[:, :args.curriculum_seqlen].contiguous()
        position_ids = position_ids[:, :args.curriculum_seqlen].contiguous()
        if labels is not None:
            labels = labels[:, :args.curriculum_seqlen].contiguous()
        loss_mask = loss_mask[:, :args.curriculum_seqlen].contiguous()

    return (tokens, position_ids, attention_mask), (labels, loss_mask)

## Code from megatron.training.py
'''
Since v0.9.0, deepspeed.initialize() has forbidden simultaneous setting of args.deepspeed_config (Path) and ds_config dict.
So, we use ds_config dict which is the more flexible option. 
'''
def _create_ds_config_dict():
    args = get_args()
    with open(args.deepspeed_config, 'r', encoding='utf-8') as config_file:
        ds_config_dict = json.load(config_file)

    if args.universal_checkpoint:
        ds_config_dict["checkpoint"] = {"load_universal": True}

    # Clear config path
    args.deepspeed_config = None 

    return ds_config_dict

def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x

def build_test_data_iterators(test_dataloader):
    """Build pretraining data iterators. Adapted to make only the test dataset given the loader"""

    args = get_args()

    # Build iterators.
    dl_type = args.dataloader_type
    assert dl_type in ['single', 'cyclic']

    test_data_iterator = iter(test_dataloader) if dl_type == 'single' \
                            else iter(cyclic_iter(test_dataloader))

    return test_data_iterator

## Code for building the test dataset. Adapted from megatron.data.gpt_dataset.py
def get_test_split_(splits_string, size):
    """
    Get the test dataset splits from comma or '/' separated string list.
    
    Args:
        splits_string (str): The splits string. (e.g. 0.99, 0.005, 0.005)
        size (int): The size of the test dataset.
    
    Returns:
        list: Returns the index of the first element of the test dataset depending on the splits argument and the
         index of the last element of the test dataset (which should be the last element of the dataset).
    """

    splits = []
    # Parse the splits string.
    if splits_string.find(',') != -1:
        splits = [float(s) for s in splits_string.split(',')]
    elif splits_string.find('/') != -1:
        splits = [float(s) for s in splits_string.split('/')]
    else:
        splits = [float(splits_string)]
    
    # We suppose that if splits_string is a single float, it is the proportion of the training set
    # thus, we fill the splits list with 0.0 for the validation and test set
    while len(splits) < 3:
        splits.append(0.)
    # If the splits list has more than 3 elements, we keep only the first 3 corresponding to the training,
    # validation and test set
    splits = splits[:3]

    # Normalize the splits.
    splits_sum = sum(splits)
    assert splits_sum > 0.0
    splits = [split / splits_sum for split in splits]

    # Get the splits index.
    splits_index = [0]
    for index, split in enumerate(splits):
        splits_index.append(splits_index[index] +
                            int(round(split * float(size))))
    # Adjust the splits index if it includes zero samples
    if splits[1] > 0 and splits_index[2] - splits_index[1] < 1:
        if splits_index[1] > 1:
            splits_index[1] -= 1
            splits_index[2] = splits_index[1] + 1

    if splits[2] > 0 and splits_index[3] - splits_index[2] < 1:
        if splits_index[2] - splits_index[1] > 1:
            splits_index[2] -= 1
            splits_index[3] = splits_index[2] + 1
        else:
            if splits_index[1] > 1:
                splits_index[1] -= 1
                splits_index[2] -= 1
                splits_index[3] = splits_index[2] + 1
    diff = splits_index[-1] - size
    for index in range(1, len(splits_index)):
        splits_index[index] -= diff
    
    # We should have indexes for the three datasets (train, valid, test) but we only need the last two 
    # (start test index, end test index)
    assert len(splits_index) == 4
    assert splits_index[-1] == size
    return splits_index[-2:]

def parse_dataprefix(data_prefix):
    """
    Parse the data prefix string. We expect the following format: [weight1, path1, weight2, path2, ...]
    
    Args:
        data_prefix (str): The data prefix string.
    
    Returns:
        list: The list of prefixes. (e.g. [path1, path2, ...])
    """
    assert len(data_prefix) % 2 == 0
    num_datasets = len(data_prefix) // 2
    prefixes = [0]*num_datasets
    for i in range(num_datasets):
        prefixes[i] = (data_prefix[2*i+1]).strip()
    return prefixes

def build_test_datasets(data_prefix, data_impl, splits_string,
                                    seq_length, seed, skip_warmup,
                                    return_doc_ids=False, *,
                                    data_cache_path=None):
    """
    Build the list of test datasets corresponding to the data prefix and the proportion of test.

    Args:
        data_prefix (str): The data prefix string.
        data_impl (str): The data implementation.
        splits_string (str): The splits string.
        seq_length (int): The sequence length of inputs.
        seed (int): The seed for the random number generator.
        skip_warmup (bool): Skip the warmup.
        return_doc_ids (bool): Return the document ids.
        data_cache_path (str): The path to the data cache for the corresponding test datasets for faster reloading.

    Returns:
        list: The list of test datasets.
    """
    print_rank_0("Start building test datasets.")

    # Single dataset.
    if len(data_prefix) == 1:
        print_rank_0(f' > prefix: {data_prefix[0]}')
        return _build_test_datasets(data_prefix[0],
                                    data_impl, splits_string,
                                    seq_length, seed, skip_warmup,
                                    data_cache_path=data_cache_path)

    # Parse the values.
    prefixes = parse_dataprefix(data_prefix)
    print_rank_0(f' > prefixes: {prefixes}')

    # Build individual datasets.
    test_datasets = []
    for i in range(len(prefixes)):
        print_rank_0(f' > building test dataset: {prefixes[i]}')
        test_ds = _build_test_datasets(
            prefixes[i], data_impl, splits_string,
            seq_length, seed, skip_warmup,
            return_doc_ids,
            data_cache_path=data_cache_path)

        test_datasets.append(test_ds)

    return test_datasets

def _build_test_datasets(data_prefix, data_impl, splits_string,
                                     seq_length, seed, skip_warmup,
                                     return_doc_ids=False, *,
                                     data_cache_path=None):
    """Build test datasets. Adapted from gpt_dataset.py to return only the test dataset."""

    # Indexed dataset.
    indexed_dataset = get_indexed_dataset_(data_prefix,
                                           data_impl,
                                           skip_warmup)

    total_num_of_documents = indexed_dataset.sizes.shape[0]
    test_split = get_test_split_(splits_string, total_num_of_documents)
    test_num_samples = test_split[1] - test_split[0]
    # Print stats about the splits.
    print_rank_0('     document indices in [{}, {}) total of {} '
                    'documents'.format(test_split[0], test_split[1],
                                        test_num_samples))

    test_dataset = None
    if test_split[1] > test_split[0]:
        documents = np.arange(start=test_split[0], stop=test_split[1],
                                step=1, dtype=np.int32)
        test_dataset = GPTDataset(f"{data_prefix}", data_prefix, documents, indexed_dataset,
                                splits_string,
                                test_num_samples,
                                seq_length, seed,
                                return_doc_ids,
                                data_cache_path=data_cache_path)

    return test_dataset

def get_test_dataset_args(parser):
    """
    Provide extra arguments required for generating the test dataset and saving the results.
    """
    group = parser.add_argument_group(title='Perlexity evaluation arguments')
    group.add_argument('--skip-warmup', type=bool, default=False)
    group.add_argument('--datatest-cache-path', type=str, default=None)
    group.add_argument('--perplexity-results-path', type=str, default=None)
    return parser

### Keep it as it may be useful later
# def build_data_loader(dataset, micro_batch_size, num_workers, drop_last,
#         task_collate_fn=None):
#     """Data loader. Note that batch-size is the local (per GPU) batch-size."""

#     # Sampler.
#     world_size = mpu.get_data_parallel_world_size()
#     rank = mpu.get_data_parallel_rank()
#     sampler = torch.utils.data.distributed.DistributedSampler(
#         dataset, num_replicas=world_size, rank=rank)

#     # Data loader. Note that batch size is the per GPU batch size.
#     data_loader = torch.utils.data.DataLoader(dataset,
#                                               batch_size=micro_batch_size,
#                                               sampler=sampler,
#                                               shuffle=False,
#                                               num_workers=num_workers,
#                                               drop_last=drop_last,
#                                               pin_memory=True,
#                                               collate_fn=task_collate_fn)

#     return data_loader

def build_pretraining_data_loader(dataset, consumed_samples):
    """Buld dataloader given an input dataset."""

    if dataset is None:
        return None
    args = get_args()

    batch_sampler = MegatronPretrainingSampler(
        total_samples=len(dataset),
        consumed_samples=consumed_samples,
        micro_batch_size=args.micro_batch_size,
        data_parallel_rank=mpu.get_data_parallel_rank(),
        data_parallel_size=mpu.get_data_parallel_world_size(),
        drop_last=False
        )

    # Torch dataloader.
    return torch.utils.data.DataLoader(dataset,
                                       batch_sampler=batch_sampler,
                                       num_workers=args.num_workers,
                                       pin_memory=True)

# Rewrite the evaluate of megatron.training.py with only the necessary arguments
def evaluate(data_iterator,
             model,
             eval_iters,
             verbose=True,
            ):
    """Return the total loss per token on the given data loader."""
    args = get_args()

    # Turn on evaluation mode which disables dropout.
    for model_module in model:
        model_module.eval()

    total_loss_dict = {}

    with torch.no_grad():
        # NB: This eval iters here should be the number of microbatches
        # times the number of processes running simultaneously but hard to get
        # so for now we just use a very large number and use the try except to break 
        for iteration in range(1, eval_iters + 1):
            if verbose and iteration % 100 == 0:
                print_rank_0('Evaluating iter {}/{}'.format(iteration,
                                                            eval_iters))

            # DeepSpeed uses eval_batch() and already average losses. Other reductions don't work so we have to manually
            # get the sum of the losses instead of the average
            try:
                loss = model[0].eval_batch(data_iterator) # average loss per sample per microbatch
                # difficult to know if it is the right way to get the total loss
                loss = loss * args.micro_batch_size * args.seq_length # losses per token
                loss_dicts = [{'lm loss' : loss}]
            except StopIteration:
                break

            # Empty unused memory
            if args.empty_unused_memory_level >= 1:
                torch.cuda.empty_cache()

            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                # Reduce across processes.
                for loss_dict in loss_dicts:
                    for key in loss_dict:
                        if 'moe' not in key:
                            total_loss_dict[key] = total_loss_dict.get(
                                key, get_accelerator().FloatTensor([0.0])) + loss_dict[key]

    # Move model back to the train mode.
    for model_module in model:
        model_module.train()

    return total_loss_dict


def evaluate_and_print_results(data_loader, model, num_original_tokens, num_tokenized_tokens, eval_iters):
    """Evaluate, print results on screen and save them."""
    print_rank_0('Evaluating ...')
    total_loss_dict = evaluate(data_loader, model, eval_iters)
    print_rank_0(f"total_loss_dict: {total_loss_dict}")
    string = ''

    # Compute the metrics
    val_loss = torch.tensor(total_loss_dict['lm loss'].item() / (num_tokenized_tokens - 1),
                            device=get_accelerator().device_name())
    ppl = torch.tensor(math.exp(min(20, val_loss.item())), device=get_accelerator().device_name())
    token_ratio = torch.tensor((num_tokenized_tokens - 1) / (num_original_tokens - 1),
                               device=get_accelerator().device_name())
    adjusted_ppl = torch.tensor(math.exp(min(20, val_loss.item() * token_ratio.item())),
                                device=get_accelerator().device_name())

    # All reduce to synchronize the results across GPUs
    dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)  # mean reduction is not supported
    dist.all_reduce(ppl, op=dist.ReduceOp.SUM)
    dist.all_reduce(adjusted_ppl, op=dist.ReduceOp.SUM)
    dist.all_reduce(token_ratio, op=dist.ReduceOp.SUM)

    # Divide by data shards
    NB_SHARDS = mpu.get_data_parallel_world_size()
    print_rank_0(f" > NB_SHARDS: {NB_SHARDS}")
    val_loss = val_loss / NB_SHARDS
    token_ratio = token_ratio / NB_SHARDS

    # Print some results
    string += 'avg loss: {:.4E} | '.format(val_loss)
    string += 'ppl: {:.4E} | '.format(ppl)
    string += 'adjusted ppl: {:.4E} | '.format(adjusted_ppl)
    string += 'token ratio: {} |'.format(token_ratio)

    # Save in dictionary for one domain
    results = {
        "loss": val_loss.detach().cpu().numpy(),
        "ppl": ppl.detach().cpu().numpy(),
        "adjusted_ppl": adjusted_ppl.detach().cpu().numpy(),
        "token_ratio": token_ratio.detach().cpu().numpy()
    }

    length = len(string) + 1
    print_rank_0('-' * length)
    print_rank_0(string)
    print_rank_0('-' * length)

    return results


### Main
def main():
    args = get_args()
    tokenizer = get_tokenizer()

    test_datasets = build_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=args.skip_warmup,
        data_cache_path=args.datatest_cache_path
    )

    model_type = ModelType.encoder_or_decoder
    args.deepspeed_config_dict = _create_ds_config_dict()
    model, _, __ = setup_model_and_optimizer(
        model_provider, model_type, teacher=False, data_post_process=None,
        build_train_valid_test_datasets_provider=None)

    # Use the datasets to evaluate the model
    domain_ppls = {}
    nb_ds = len(test_datasets)
    for idx, ds in enumerate(test_datasets):
        print_rank_0(f"Dataset #{idx}/{nb_ds}")
        dataloader = build_pretraining_data_loader(ds, 0) # 0 is the number of consumed samples
        num_tokenized_tokens = 0
        num_original_tokens = 0
        eval_iters = 0
        for batch in dataloader:
            tokens_ = batch['text'].long().to(get_accelerator().device_name()).contiguous()
            text_ = []
            for tok in tokens_:
                text_.append(tokenizer.detokenize(tok))
            num_original_tokens += sum([len(t.strip().split(" ")) for t in text_])
            num_tokenized_tokens += sum([len(tokenizer.tokenize(t)) for t in text_])
            eval_iters += 1
        print_rank_0(f"eval iters: {eval_iters}")
        print_rank_0(f"num_original_tokens: {num_original_tokens}")
        print_rank_0(f"num_tokenized_tokens: {num_tokenized_tokens}")
        test_iter = build_test_data_iterators(dataloader)

        # Evaluate the model
        results = evaluate_and_print_results(test_iter, model, num_original_tokens, num_tokenized_tokens, eval_iters)

        # Fetch the results
        ppl = results['ppl']
        loss = results['loss']
        adjusted_ppl = results['adjusted_ppl']
        token_ratio = results['token_ratio']
        print(f"Dataset name: {ds.name} | Loss: {loss:.5f} | PPL: {ppl:.5f} | Adjusted PPL: {adjusted_ppl:.5f} | Token Ratio: {token_ratio:.5f}")

        # Store them in a dictionary
        domain_ppls[ds.name] = {
            "loss": loss,
            "ppl": ppl,
            "adjusted_ppl": adjusted_ppl,
            "num_original_tokens": num_original_tokens,
            "num_tokenized_tokens": num_tokenized_tokens,
            "token_ratio": token_ratio,
        }

    # Now dump the results to a csv file
    path = args.perplexity_results_path

    if not os.path.exists(path):
        os.makedirs(path)

    with open(f"{path}/perplexity.csv", mode='w') as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset_name", "Loss", "PPL", "Adjusted_PPL", "Num_original_tokens", "Num_tokenized_tokens", "Token_Ratio"])
        for ds_name, ds_data in domain_ppls.items():
            loss = ds_data['loss']
            ppl = ds_data['ppl']
            adj_ppl = ds_data['adjusted_ppl']
            nb_origin_tok = ds_data['num_original_tokens']
            nb_tok_tok = ds_data['num_tokenized_tokens']
            tok_ratio = ds_data['token_ratio']
            writer.writerow([ds_name, loss, ppl, adj_ppl, nb_origin_tok, nb_tok_tok, tok_ratio])


if __name__ == "__main__":
    initialize_megatron(extra_args_provider=get_test_dataset_args)
    args = get_args()
    print_rank_0(f"Arguments have been initialized:{args}")
    main()
