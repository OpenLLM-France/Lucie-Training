# This script is inspired from the original script of the Croissant team :
# https://github.com/ManuelFay/llm-data-hub/blob/ea9c84708f00f61320ea352998e6af999aa71c24/dataset_construction/fit_tokenizer.py

import itertools
import json
import re
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

_space_internal = "▁"


def build_tokenizer(
    dropout: Optional[float] = None,
    individual_digits: Optional[bool] = True,
    separate_spaces_and_punctuations: Optional[bool] = True,
    space_behaviour: Optional[str] = "prefix_all",
    fuse_unk: Optional[float] = True,
    do_not_split_spaces: Optional[bool] = False,
    char_to_prefix_space: Optional[str] = "\n\t(['\"«“‘‚‹—–―",
):
    """
    Build a tokenizer.
    :return: The tokenizer.

    :param dropout: Dropout rate for BPE
    :param individual_digits: Split digits individually
    :param separate_spaces_and_punctuations: Make sure not to mix spaces and punctuations with alphanumeric characters
    :param space_behaviour:
        - "prefix_sos": Add a prefix space at the start of the text
        - "prefix_all": Add a prefix space after each linebreaks and tabulations (not just after the start)
        - "split": Do not mix space with other characters
    :param fuse_unk: Fuse unknown tokens
    :param do_not_split_spaces: Experimental (not working)
    """

    assert space_behaviour in {"prefix_sos", "prefix_all", "split"}

    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.BPE(
            dropout=dropout,
            unk_token="<unk>",  # None,
            fuse_unk=fuse_unk,
            byte_fallback=True,
        )
    )

    add_prefix_space = space_behaviour != "split"

    normalizers = [
        tokenizers.normalizers.NFKC(),  # Note: This replaces unbreakable space "\u00A0" -> " "
        tokenizers.normalizers.Replace("\r", ""),
    ] + (
        [
            # Note: placeholders cannot be handled (using tokenizers.Regex(r"[\t\n]"))
            tokenizers.normalizers.Replace(tokenizers.Regex(rf"({re.escape(c)})(?=[^\1])"), c + " ")
            for c in char_to_prefix_space
        ]
        if (space_behaviour == "prefix_all")
        else ([tokenizers.normalizers.Replace(" ", _space_internal)] if space_behaviour == "split" else [])
    )

    pretokenizers = (
        [
            # V1
            # tokenizers.pre_tokenizers.Split(
            #     tokenizers.Regex(rf"{_space_internal}?([\n\t\p{{P}}])\1*"), behavior="isolated"
            # ),
            # V2
            tokenizers.pre_tokenizers.Split(tokenizers.Regex(rf"{_space_internal}?\p{{P}}+"), behavior="isolated"),
            tokenizers.pre_tokenizers.Split(tokenizers.Regex(r"[\n\t]"), behavior="isolated"),
        ]
        if (separate_spaces_and_punctuations and space_behaviour != "split")
        else (
            (
                [
                    tokenizers.pre_tokenizers.Split(tokenizers.Regex(rf"[{_space_internal}\n\t]"), behavior="isolated"),
                ]
                + (
                    [tokenizers.pre_tokenizers.Split(tokenizers.Regex(r"\p{P}+"), behavior="isolated")]
                    if separate_spaces_and_punctuations
                    else []
                )
            )
            if space_behaviour == "split"
            else []
        )
    ) + [
        tokenizers.pre_tokenizers.Digits(individual_digits=individual_digits),
    ]

    if not do_not_split_spaces:
        tokenizer.normalizer = tokenizers.normalizers.Sequence(normalizers)

        tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Sequence(
            (
                (
                    [
                        tokenizers.pre_tokenizers.Metaspace(
                            replacement=_space_internal, add_prefix_space=add_prefix_space
                        ),
                    ]
                )
                if space_behaviour != "split"
                else []
            )
            + pretokenizers
        )

    else:  # Mistral + digits
        tokenizer.normalizer = tokenizers.normalizers.Sequence(
            normalizers
            + [
                tokenizers.normalizers.Replace(" ", _space_internal),
                tokenizers.normalizers.Prepend(_space_internal),
            ]
        )

        tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Sequence(pretokenizers)

    tokenizer.decoder = tokenizers.decoders.Sequence(
        [
            # tokenizers.decoders.Replace("▁", " "),
            # tokenizers.decoders.ByteFallback(),
            tokenizers.decoders.ByteFallback(),
            tokenizers.decoders.Metaspace(replacement=_space_internal, add_prefix_space=add_prefix_space),
            tokenizers.decoders.Fuse(),
        ]
        + (
            [tokenizers.decoders.Replace(c + " ", c) for c in char_to_prefix_space]
            if (space_behaviour == "prefix_all")
            else []
        )
        + [
            # tokenizers.decoders.Strip(content=" ", left=1, right=0),
        ]
    )

    return tokenizer


def get_special_tokens(
    num_unused=40,
    consecutive_spaces=8,
    consecutive_tabs=4,
    consecutive_linebreaks=2,
    special_tokens_map=None,
):
    if special_tokens_map is None:
        special_tokens_map = {}
    new_special_tokens = []
    for key, value in _special_tokens_map.items():
        expected_content = value["content"]
        if key in special_tokens_map:
            content = special_tokens_map[key]
            if special_tokens_map[key] != expected_content:
                raise NotImplementedError(
                    f"Changing special tokens is not supported yet. {key} is {content} and should be {expected_content}"
                )
        else:
            new_special_tokens.append(expected_content)

    # For byte fallback
    new_special_tokens += [f"<0x{i:02X}>" for i in range(256)]

    new_special_tokens += [f"<unused{i}>" for i in range(num_unused)]
    for n, char in (
        (consecutive_spaces, _space_internal),
        (consecutive_tabs, "\t"),
        (consecutive_linebreaks, "\n"),
    ):
        for i in range(1, n + 1):
            new_special_tokens.append(char * i)

    for i in range(10):
        new_special_tokens.append(f"{i}")

    return new_special_tokens


def fit_tokenizer(tokenizer, it, len_it=None, vocab_size=32000, batch_size=1000, **special_tokens_options):
    """
    Fit a tokenizer on a dataset.
    :param tokenizer: The tokenizer to fit.
    :param it: Generator of texts.
    :param len_it: Length of the generator (optional, only used for the progress bar).
    :param vocab_size: Size of the vocabulary.
    :param batch_size: Size of the batches.
    :return: The fitted tokenizer.
    """

    special_tokens = get_special_tokens(**special_tokens_options)

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
    **special_tokens_options,
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

    new_special_tokens = get_special_tokens(special_tokens_map=tokenizer.special_tokens_map, **special_tokens_options)

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


def add_consecutive_spaces(  # noqa # C901 `...` is too complex
    tokenizer_file,
    consecutive_spaces=8,
    consecutive_tabs=4,
    consecutive_linebreaks=2,
):
    tokenizer = json.load(open(tokenizer_file, encoding="utf8"))
    tokens = tokenizer["model"]["vocab"]

    # Add consecutive spaces in the vocabulary
    # n = 2
    # assert n <= max_length
    # for token, val in tokens.copy().items():
    #     if token.startswith("<unused"):
    #         while "▁" * n in tokens.keys():
    #             n += 1
    #         if n > max_length:
    #             break
    #         tokens.pop(token)
    #         tokens["▁" * n] = val
    #         n += 1
    #     elif n > 2:
    #         break
    # # Re-sort tokens
    # tokenizer["model"]["vocab"] = dict(sorted(tokens.items(), key=lambda x: x[1]))

    # Add consecutive spaces in the merges
    for n, char in (
        (consecutive_spaces, _space_internal),
        (consecutive_tabs, "\t"),
        (consecutive_linebreaks, "\n"),
    ):
        # Make all the possible combinations
        all_spaces = [char * i for i in range(1, n)]

        # Check that all the spaces are in the tokens
        for s in all_spaces:
            assert s in tokens, f"{s} not in tokens"
        assert (char * n) in tokens, f"{char * n} not in tokens"
        assert (char * (n + 1)) not in tokens, f"{char * (n+1)} in tokens"

        # Safety (make sure all merges produce valid tokens)
        all_spaces = [s for s in all_spaces if s in tokens]

        if len(all_spaces) < 2:
            continue

        all_pairs = list(itertools.product(all_spaces, repeat=2))
        new_merges = sorted(
            [f"{a} {b}" for a, b in all_pairs],
            # key=len
            key=lambda x: (-len(x.split(" ")[0]), -len(x.split(" ")[1])),
        )
        new_merges = [m for m in new_merges if m.replace(" ", "") in all_spaces]
        merges = tokenizer["model"]["merges"]
        tokenizer["model"]["merges"] = [m for m in merges if m not in new_merges] + new_merges

    # Replace "Metaspace" pre_tokenizer by "Replace" normalizer
    pre_tokenizer = tokenizer["pre_tokenizer"]
    isseq_pre_tokenizer = pre_tokenizer["type"] == "Sequence"
    has_metaspace = (
        ("Metaspace" in [p["type"] for p in pre_tokenizer["pretokenizers"]])
        if isseq_pre_tokenizer
        else (pre_tokenizer["type"] == "Metaspace")
    )
    if has_metaspace:
        add_prefix_space = (
            [p["add_prefix_space"] for p in pre_tokenizer["pretokenizers"] if p["type"] == "Metaspace"][0]
            if isseq_pre_tokenizer
            else pre_tokenizer["add_prefix_space"]
        )

        # Add Replace in the normalizer
        normalizer = tokenizer["normalizer"]
        isseq_normalizer = normalizer["type"] == "Sequence"
        new_normalizers = ([{"type": "Prepend", "prepend": _space_internal}] if add_prefix_space else []) + [
            {"type": "Replace", "pattern": {"String": " "}, "content": _space_internal},
        ]
        if isseq_normalizer:
            normalizer["normalizers"] += new_normalizers
        else:
            normalizer = {
                "type": "Sequence",
                "normalizers": [normalizer] + new_normalizers,
            }
        tokenizer["normalizer"] = normalizer
        # Remove Metaspace from the pre_tokenizer
        if isseq_pre_tokenizer:
            pre_tokenizer = {
                "type": "Sequence",
                "pretokenizers": [p for p in pre_tokenizer["pretokenizers"] if p["type"] != "Metaspace"],
            }
        else:
            pre_tokenizer = None

    # Remove pre_tokenizer that will be useless afterwards
    for type in (
        "Digits",
        "Split",
    ):
        has_pretokenizer = (
            (type in [p["type"] for p in pre_tokenizer["pretokenizers"]])
            if isseq_pre_tokenizer
            else (pre_tokenizer["type"] == type)
        )
        if has_pretokenizer:
            # Remove Split from the pre_tokenizer
            if isseq_pre_tokenizer:
                pre_tokenizer = {
                    "type": "Sequence",
                    "pretokenizers": [p for p in pre_tokenizer["pretokenizers"] if p["type"] != type],
                }
            else:
                pre_tokenizer = None

    if isseq_pre_tokenizer:
        if pre_tokenizer["pretokenizers"] == []:
            pre_tokenizer = None
        elif len(pre_tokenizer["pretokenizers"]) == 1:
            pre_tokenizer = pre_tokenizer["pretokenizers"][0]
    tokenizer["pre_tokenizer"] = pre_tokenizer

    json.dump(
        tokenizer,
        open(tokenizer_file, "w", encoding="utf8"),
        indent=2,
        ensure_ascii=False,
    )


if __name__ == "__main__":
    import argparse
    import os
    import shutil
    import sys
    import time

    def str2bool(s):
        s = s.lower()
        assert s in {"true", "false", "1", "0"}
        return s.lower() in {"true", "1"}

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
        "--individual_digits",
        default=True,
        type=str2bool,
        help="Split digits individually (ex: 1999 -> 1 9 9 9)",
    )
    parser.add_argument(
        "--consecutive_spaces",
        default=8,
        type=int,
        help="Maximum number of consecutive spaces (in a same token)",
    )
    parser.add_argument(
        "--consecutive_tabs",
        default=4,
        type=int,
        help="Maximum number of consecutive tabs (in a same token)",
    )
    parser.add_argument(
        "--consecutive_linebreaks",
        default=2,
        type=int,
        help="Maximum number of consecutive linebreaks (in a same token)",
    )
    parser.add_argument(
        "--space_behaviour",
        default="prefix_all",
        choices=["prefix_all", "prefix_sos", "split"],
        help="How to deal with whitespaces",
    )
    parser.add_argument(
        "--separate_spaces_and_punctuations",
        default=False,
        type=str2bool,
        help="Make sure not to mix spaces and punctuations with alphanumeric characters",
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
    args = parser.parse_args()

    if args.verbose:
        print("Configure base tokenizer")

    name_tokenizer = os.path.basename(args.base) if args.base else "Lucie"
    name_tokenizer += f"-{args.vocab_size}"
    name_tokenizer += f"-sp{args.consecutive_spaces}-{args.consecutive_tabs}-{args.consecutive_linebreaks}"
    if not args.base:
        if args.individual_digits:
            name_tokenizer += "-digits"
        if args.separate_spaces_and_punctuations:
            name_tokenizer += "-punctsV2"
        name_tokenizer += f"-{args.space_behaviour.replace('_', '')}"

    example_sentence = (
        "   [INST] Coucou [/INST] Hello [INST] "
        "Mais en Français, comment est-ce que ça se passera ? 1999, 2000, 2002.2, 1 000 000, 1\u00A0000\u00A0000"
        "[/INST] Eh bien ␣ alors あ゙"
    )

    info = {}

    if args.debug:
        name_dataset = "dummy"
        example_training_sentences = [
            "   Mais en Français, comment   est-ce que ça se passera?\n\ns hey... ow ...?",
            "1999 2000 1999   2000 199 200 en François en Français\n\ns ra? ow...? hey...",
            "sans oublier Mot (Mot) [Mot] (Mot) [Mot]",
        ]

        def debug_texts_iterator():
            yield from example_training_sentences

        trainset = debug_texts_iterator()
    else:
        trainset = tokenizer_dataset(
            train=True,
            streaming=True,
        )
        name_dataset = trainset.name

    if not args.output:
        args.output = f"trained/tokenizer_{name_tokenizer}_{name_dataset}"
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
    # Self-copy the script
    shutil.copy2(__file__, os.path.join(args.output, os.path.basename(__file__)))

    # Compute and dump training stats in parallel
    if not args.debug:
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
        tok = transformers.AutoTokenizer.from_pretrained(args.base)
        tok = refit_tokenizer(
            tok,
            trainset,
            consecutive_spaces=args.consecutive_spaces,
            consecutive_tabs=args.consecutive_tabs,
            consecutive_linebreaks=args.consecutive_linebreaks,
        )
        tok.save_pretrained(args.output)

    else:
        # From scratch
        tok = build_tokenizer(
            individual_digits=args.individual_digits,
            separate_spaces_and_punctuations=args.separate_spaces_and_punctuations,
            space_behaviour=args.space_behaviour,
        )

        # Print options and stress test
        print(json.dumps(json.loads(tok.to_str())["normalizer"], indent=2))
        print(json.dumps(json.loads(tok.to_str())["pre_tokenizer"], indent=2))
        for s in [
            "123 456\u00A0789",
            "Mot.Mot. Mot...  Mot (Mot) (Mot)   (Mot) \nMot \n Mot",
        ]:
            print(s.replace("\n", "\\n"))
            tokens = tok.pre_tokenizer.pre_tokenize_str(tok.normalizer.normalize_str(s))
            tokens = [t[0] for t in tokens]
            print(tokens)

        tok = fit_tokenizer(
            tok,
            trainset,
            consecutive_spaces=args.consecutive_spaces,
            consecutive_tabs=args.consecutive_tabs,
            consecutive_linebreaks=args.consecutive_linebreaks,
        )
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

    add_consecutive_spaces(
        os.path.join(args.output, "tokenizer.json"),
        consecutive_spaces=args.consecutive_spaces,
        consecutive_tabs=args.consecutive_tabs,
        consecutive_linebreaks=args.consecutive_linebreaks,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.output)

    # Launch evaluation
    if not args.debug:
        for dataset in [
            "Wikipedia:fr",
            "Wikipedia:en",
            "Wikipedia:de",
            "Wikipedia:es",
            "Wikipedia:it",
            "Europarl",
            "Gutenberg",
            "TheStack",
            "Persee",
            "Gallica",
        ]:
            os.system(
                f"""\
{sys.executable} {os.path.dirname(os.path.realpath(__file__))}/tokenizer_eval.py {args.output} --regex {dataset} &
"""
            )

    else:
        for sentence in example_training_sentences:
            tokens, decoded = test_tokenizer(tokenizer, sentence)
            print("* Input:  ", sentence.replace("\n", "\\n"))
            print("* Tokens: ", tokens)
            print("* Decoded:", decoded.replace("\n", "\\n"))
            print()

        os.system(
            f"""\
{sys.executable} {os.path.dirname(os.path.realpath(__file__))}/tokenizer_quicktest.py {args.output}
"""
        )
