# This script is inspired from the original script of the Croissant team :
# https://github.com/ManuelFay/llm-data-hub/blob/ea9c84708f00f61320ea352998e6af999aa71c24/dataset_construction/fit_tokenizer.py

from typing import Optional

import tokenizers
import transformers

from data import tokenizer_dataset

_special_tokens_map = {
    "bos_token": {
        "content": "<s>",
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
    },
    "eos_token": {
        "content": "</s>",
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
    },
    "unk_token": {
        "content": "<unk>",
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
    },
    "pad_token": {
        "content": "<pad>",
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
    },
}

_special_tokens = [
    s["content"] for s in [_special_tokens_map[k] for k in ["unk_token", "bos_token", "eos_token", "pad_token"]]
]


def build_tokenizer(
    dropout: Optional[float] = None,
    fuse_unk: Optional[float] = True,
    same_as_croissant_llm: Optional[bool] = True,
):
    """
    Build a tokenizer.
    :return: The tokenizer.
    """
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.BPE(
            dropout=dropout,
            unk_token="<unk>",  # None,
            fuse_unk=fuse_unk,
            byte_fallback=True,
        )
    )
    if same_as_croissant_llm:
        add_prefix_space = True
        replacement = "▁"

        tokenizer.normalizer = tokenizers.normalizers.NFKC()

        tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Sequence(
            [
                tokenizers.pre_tokenizers.Metaspace(replacement=replacement, add_prefix_space=add_prefix_space),
                tokenizers.pre_tokenizers.Digits(individual_digits=True),
            ]
        )

        tokenizer.decoder = tokenizers.decoders.Sequence(
            [
                tokenizers.decoders.ByteFallback(),
                tokenizers.decoders.Metaspace(replacement=replacement, add_prefix_space=add_prefix_space),
                tokenizers.decoders.Fuse(),
                tokenizers.decoders.Strip(content=" ", left=1, right=0),
            ]
        )
    else:  # Mistral + digits
        tokenizer.normalizer = tokenizers.normalizers.Sequence(
            [
                tokenizers.normalizers.NFKC(),
                tokenizers.normalizers.Prepend("▁"),
                tokenizers.normalizers.Replace(" ", "▁"),
            ]
        )

        tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Digits(individual_digits=True)

        tokenizer.decoder = tokenizers.decoders.Sequence(
            [
                # tokenizers.decoders.Replace("▁", " "),
                # tokenizers.decoders.ByteFallback(),
                tokenizers.decoders.ByteFallback(),
                tokenizers.decoders.Metaspace(replacement="▁", add_prefix_space=True),
                tokenizers.decoders.Fuse(),
                tokenizers.decoders.Strip(content=" ", left=1, right=0),
            ]
        )

    return tokenizer


def fit_tokenizer(
    tokenizer,
    it,
    len_it=None,
    vocab_size=32000,
    batch_size=1000,
):
    """
    Fit a tokenizer on a dataset.
    :param tokenizer: The tokenizer to fit.
    :param it: Generator of texts.
    :param len_it: Length of the generator (optional, only used for the progress bar).
    :param vocab_size: Size of the vocabulary.
    :param batch_size: Size of the batches.
    :return: The fitted tokenizer.
    """

    special_tokens = _special_tokens.copy()

    special_tokens += [f"<unused{i}>" for i in range(10)]

    # For byte fallback
    special_tokens += [f"<0x{i:02X}>" for i in range(256)]

    # special_tokens += [f"<extra_id_{i}>" for i in range(100)]
    bpe_trainer = tokenizers.trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True,
        special_tokens=special_tokens,
        limit_alphabet=1000,
        initial_alphabet=[],
    )
    tokenizer.train_from_iterator(batchify_iterator(it, batch_size=batch_size), trainer=bpe_trainer, length=len_it)
    return tokenizer


def refit_tokenizer(
    tokenizer: transformers.PreTrainedTokenizerFast,
    it,
    len_it=None,
    vocab_size=32000,
    batch_size=1000,
):
    """
    Fit a tokenizer on a dataset.
    :param tokenizer: The tokenizer to fit.
    :param it: Generator of texts.
    :param len_it: Length of the generator (optional, only used for the progress bar).
    :param vocab_size: Size of the vocabulary.
    :param batch_size: Size of the batches.
    :return: The fitted tokenizer.
    """

    # is this slowing down everything ?

    new_special_tokens = []
    for key, value in _special_tokens_map.items():
        expected_content = value["content"]
        if key in tokenizer.special_tokens_map:
            content = tokenizer.special_tokens_map[key]
            if tokenizer.special_tokens_map[key] != expected_content:
                raise NotImplementedError(
                    f"Changing special tokens is not supported yet. {key} is {content} and should be {expected_content}"
                )
        else:
            new_special_tokens.append(expected_content)

    new_special_tokens += [f"<unused{i}>" for i in range(10)]

    # For byte fallback
    new_special_tokens += [f"<0x{i:02X}>" for i in range(256)]

    tokenizer._tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Digits(individual_digits=True)

    tokenizer = tokenizer.train_new_from_iterator(
        batchify_iterator(it, batch_size=batch_size),
        vocab_size=vocab_size,
        length=len_it,
        new_special_tokens=new_special_tokens,
    )

    return tokenizer


def batchify_iterator(it, batch_size=1000):
    """
    Batchify an iterator.
    :param it: The iterator to batchify.
    :param batch_size: The size of the batches.
    :return: The batchified iterator.
    """
    batch = []
    for x in it:
        batch.append(x)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def test_tokenizer(tokenizer, sentence):
    if isinstance(sentence, list):
        return [test_tokenizer(tokenizer, s) for s in sentence]

    if "encode_batch" in dir(tokenizer):
        tokens = tokenizer.encode_batch([sentence])[0]
        if hasattr(tokens, "ids"):
            tokens = tokens.ids
        if hasattr(tokenizer, "id_to_token"):
            tokens_strings = [tokenizer.id_to_token(t) for t in tokens]
        else:
            tokens_strings = [tokenizer.decode([t]) for t in tokens]
    else:
        # Fast tokenizer
        # tokens = tokenizer([sentence], padding=False,
        # truncation=False)["input_ids"][0]
        tokens = tokenizer.encode(sentence, add_special_tokens=True)
        tokens_strings = tokenizer.convert_ids_to_tokens(tokens)

    return [tokens_strings, tokenizer.decode(tokens)]


def set_infinite_length(tokenizer):
    tokenizer.model_max_length = 10**30
    tokenizer.max_model_input_sizes = {}
    tokenizer.set_truncation_and_padding(
        transformers.tokenization_utils_fast.PaddingStrategy.DO_NOT_PAD,
        transformers.tokenization_utils_fast.TruncationStrategy.DO_NOT_TRUNCATE,
        10**30,
        1,
        1,
    )
    return tokenizer


if __name__ == "__main__":
    import argparse
    import json
    import os
    import shutil
    import sys
    import time

    parser = argparse.ArgumentParser(
        description="Train a tokenizer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=32000,
        help="Size of output vocabulary",
    )
    parser.add_argument(
        "--base",
        default=None,
        help="Base tokenizer (ex: mistralai/Mistral-7B-v0.1)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output folder (will be set automatically if not specified)",
    )
    parser.add_argument(
        "--overwrite",
        default=False,
        action="store_true",
        help="Overwrite output folder if it already exists",
    )
    parser.add_argument("--debug", default=False, action="store_true", help="Debug mode")
    parser.add_argument(
        "--no_verbose",
        dest="verbose",
        action="store_false",
        default=True,
    )
    data_group = parser.add_argument_group("Dataset selection")
    data_group.add_argument(
        "--no_persee",
        dest="persee",
        action="store_false",
        default=True,
        help="Don't use Persee",
    )
    data_group.add_argument(
        "--no_legi",
        dest="legi",
        action="store_false",
        default=True,
        help="Don't use LEGI",
    )
    data_group.add_argument(
        "--no_europarl",
        dest="europarl",
        action="store_false",
        default=True,
        help="Don't use Europarl",
    )
    args = parser.parse_args()

    if args.verbose:
        print("Configure base tokenizer")

    name_tokenizer = os.path.basename(args.base) if args.base else "Croissant"

    example_sentence = (
        "   [INST] Coucou [/INST] Hello [INST] "
        "Mais en Français, comment est-ce que ça se passera ? 1999, 2000, 2002.2, 1 000 000, 1\u00A0000\u00A0000"
        "[/INST] Eh bien ␣ alors あ゙"
    )

    info = {}

    if args.debug:
        name_dataset = "dummy"
        sentences = [
            "<a> Mais en Français, comment est-ce que ça se passera?",
            "1999 2000 1999 2000 199 200 en François en Français",
            "<s> zzzzzAAAA",
        ]

        def debug_texts_iterator():
            yield from sentences

        trainset = debug_texts_iterator()
    else:
        trainset = tokenizer_dataset(
            train=True,
            streaming=True,
            use_persee=args.persee,
            use_legi=args.legi,
            use_europarl=args.europarl,
        )
        name_dataset = trainset.name

    if not args.output:
        # args.output = f"trained_tokenizer_{args.tokenizer.replace('/', '--')}"
        args.output = f"trained_tokenizer_{name_tokenizer}_{name_dataset}"
        if args.debug:
            args.output = "DEBUG_" + args.output

    if args.debug:
        args.overwrite = True

    print("Output folder:", args.output)
    if os.path.exists(args.output):
        if args.overwrite:
            shutil.rmtree(args.output)
        else:
            print("WARNING: Output folder already exists, aborting")
            exit(1)
    os.makedirs(args.output)

    # Compute and dump training stats in parallel
    os.system(
        f"""\
{sys.executable} {os.path.dirname(os.path.realpath(__file__))}/data.py \
    tok_train \
    --folder {args.output}/stats_training >/dev/null 2>/dev/null &
"""
    )

    if args.verbose:
        print("Train tokenizer")

    tic = time.time()

    if args.base:
        # Refit from pretrained
        tok2 = transformers.AutoTokenizer.from_pretrained(args.base)
        tok2 = refit_tokenizer(tok2, trainset)
        tok2.save_pretrained(args.output)

    else:
        # From scratch
        tok = build_tokenizer()
        tok = fit_tokenizer(tok, trainset)
        tok.save(os.path.join(args.output, "tokenizer.json"))
        tok = transformers.PreTrainedTokenizerFast(tokenizer_file=os.path.join(args.output, "tokenizer.json"))
        tok.save_pretrained(args.output)

    training_time = time.time() - tic

    # Check tokenizer can be loaded
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.output)

    # Store training information
    if args.verbose:
        print("Evaluate tokenizer")

    info.update(
        {
            "training_time": training_time,
            "vocab_size": tokenizer.vocab_size,
            "example_after": {example_sentence: test_tokenizer(tokenizer, example_sentence)},
        }
    )

    json.dump(
        info,
        open(f"{args.output}/training_info.json", "w", encoding="utf8"),
        indent=2,
        ensure_ascii=False,
    )

    # Tune tokenizer files...
    with open(f"{args.output}/special_tokens_map.json", "w", encoding="utf8") as f:
        json.dump(_special_tokens_map, f, indent=2, ensure_ascii=False)

    tokenizer_dict = json.load(open(f"{args.output}/tokenizer.json", encoding="utf8"))
    tokenizer_dict["added_tokens"] = [t for t in tokenizer_dict["added_tokens"] if t["content"] in _special_tokens]
    with open(f"{args.output}/tokenizer.json", "w", encoding="utf8") as f:
        json.dump(
            tokenizer_dict,
            f,
            indent=2,
            ensure_ascii=False,
        )
    tokenizer_config = json.load(open(f"{args.output}/tokenizer_config.json", encoding="utf8"))
    tokenizer_config["added_tokens_decoder"] = {
        k: v
        for k, v in tokenizer_config["added_tokens_decoder"].items()
        if v["content"] in _special_tokens or "unused" in v["content"]
    }
    tokenizer_config.update(
        {
            "additional_special_tokens": [],
            "clean_up_tokenization_spaces": False,
            "add_bos_token": True,
            "add_eos_token": False,
            "bos_token": _special_tokens_map["bos_token"]["content"],
            "eos_token": _special_tokens_map["eos_token"]["content"],
            "pad_token": _special_tokens_map["pad_token"]["content"],
            "unk_token": _special_tokens_map["unk_token"]["content"],
            "model_max_length": 1000000000000000000000000000000,
            "legacy": True,
            "spaces_between_special_tokens": False,
            "tokenizer_class": (
                "LlamaTokenizer"
                if tokenizer_config.get("tokenizer_class") in [None, "PreTrainedTokenizerFast"]
                else tokenizer_config["tokenizer_class"]
            ),
            "sp_model_kwargs": {},
            "use_default_system_prompt": False,
        }
    )
    with open(f"{args.output}/tokenizer_config.json", "w", encoding="utf8") as f:
        json.dump(tokenizer_config, f, indent=2, ensure_ascii=False, sort_keys=True)

    # Check tokenizer can be loaded
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.output)

    bos_token = tokenizer.bos_token
    eos_token = tokenizer.eos_token
    tokenizer._tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
        single=f"{bos_token} $A:0 {eos_token}",
        pair=f"{bos_token} $A:0 {bos_token} $B:1 {eos_token}",
        special_tokens=[
            (bos_token, tokenizer.bos_token_id),
            (eos_token, tokenizer.eos_token_id),
        ],
    )

    tokenizer.save_pretrained(args.output)

    # Launch evaluation
    os.system(
        f"""\
{sys.executable} {os.path.dirname(os.path.realpath(__file__))}/tokenizer_eval.py {args.output} &
"""
    )
