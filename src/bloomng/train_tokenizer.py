# https://github.com/huggingface/transformers/pull/22264
# Byte Fallback Tokenizer

# https://github.com/huggingface/tokenizers/issues/1407#issue-2028070675

import re

import tokenizers
import transformers
from data import tokenizer_dataset


def make_tokenizer(
    byte_fallback=False,
):
    forced_tokens = ["INST"]

    if byte_fallback:
        name = "mistralai/Mistral-7B-v0.1"
        base_tokenizer = transformers.AutoTokenizer.from_pretrained(name)
        # Workaround to really do byte-level fallback (?)
        forced_tokens += [f"<0x{i:02X}>" for i in range(256)]

    else:
        # name = "tiiuae/falcon-7b"
        name = "gpt2"
        base_tokenizer = transformers.AutoTokenizer.from_pretrained(name)
        base_tokenizer.add_special_tokens({"bos_token": "<startoftext>"})

    # EOS / BOS
    bos_token = base_tokenizer.bos_token
    eos_token = base_tokenizer.eos_token
    base_tokenizer._tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
        single=f"{bos_token} $A {eos_token}",
        pair=f"{bos_token} $A {eos_token} $B {eos_token}",
        special_tokens=[
            (bos_token, base_tokenizer.bos_token_id),
            (eos_token, base_tokenizer.eos_token_id),
        ],
    )

    forced_tokens = [
        tokenizers.AddedToken(
            tok,
            special=True,
            normalized="<" in tok,
            single_word=False,
            rstrip=False,
            lstrip=False,
        )
        for tok in forced_tokens
    ]
    base_tokenizer.add_special_tokens({"additional_special_tokens": forced_tokens})

    assert hasattr(base_tokenizer._tokenizer, "post_processor")

    return name.replace("/", "--"), base_tokenizer


def test_tokenizer(tokenizer, sentence):
    if isinstance(sentence, list):
        return [test_tokenizer(tokenizer, s) for s in sentence]

    if "encode_batch" in dir(tokenizer):
        tokens = tokenizer.encode_batch([sentence])[0].ids
        tokens_strings = [tokenizer.id_to_token(t) for t in tokens]
    else:
        # Fast tokenizer
        # tokens = tokenizer([sentence], padding=False,
        # truncation=False)["input_ids"][0]
        tokens = tokenizer.encode(sentence, add_special_tokens=True)
        tokens_strings = tokenizer.convert_ids_to_tokens(tokens)

    return [tokens_strings, tokenizer.decode(tokens)]


if __name__ == "__main__":
    import argparse
    import json
    import os
    import shutil
    import time

    parser = argparse.ArgumentParser(description="Train a tokenizer.")
    #     parser.add_argument(
    #         "--tokenizer",
    #         type=str,
    #         default="mistralai/Mistral-7B-v0.1",
    #         help="""
    # Base tokenizer. For instance:
    # mistralai/Mistral-7B-v0.1 -> BPE with byte-level fallback.
    # meta-llama/Llama-2-7b -> BPE with byte-level fallback.
    # tiiuae/falcon-7b -> Byte-level BPE.
    # """
    #     )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=32000,
        help="Size of output vocabulary",
    )
    parser.add_argument(
        "--byte_fallback",
        default=False,
        action="store_true",
        help="Use byte-level fallback",
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
        help="Remove output folder if it already exists",
    )
    parser.add_argument(
        "--debug", default=False, action="store_true", help="Debug mode"
    )
    parser.add_argument(
        "--no_verbose",
        dest="verbose",
        action="store_false",
        default=True,
    )
    args = parser.parse_args()

    if args.verbose:
        print("Configure base tokenizer")

    name_tokenizer, base_tokenizer = make_tokenizer(args.byte_fallback)

    example_sentence = (
        f"   [INST] Coucou [/INST] Hello {base_tokenizer.eos_token} [INST] "
        f"Mais en Français, comment est-ce que ça se passera ? "
        f"[/INST] Eh bien ␣ alors あ゙"
    )

    info = {
        "example_before": {
            example_sentence: test_tokenizer(base_tokenizer, example_sentence)
        }
    }
    print(json.dumps(info, indent=2, ensure_ascii=False))

    name_dataset, trainset = tokenizer_dataset(
        train=True, streaming=True, debug=args.debug
    )

    if not args.output:
        # args.output = f"trained_tokenizer_{args.tokenizer.replace('/', '--')}"
        args.output = f"trained_tokenizer_{name_tokenizer}_{name_dataset}"
        if args.debug:
            args.output = "DEBUG_" + args.output

    if os.path.exists(args.output):
        if args.overwrite:
            shutil.rmtree(args.output)
        else:
            print(f"WARNING: Output folder already exists, aborting: {args.output}")
            exit(1)

    os.makedirs(args.output)

    if args.verbose:
        print("Train tokenizer")

    training_kwargs = {
        "vocab_size": args.vocab_size,
        "min_frequency": 0,
        "show_progress": True,
        # initial_alphabet=["▁[", "]"],
    }

    tic = time.time()
    tokenizer = base_tokenizer.train_new_from_iterator(trainset, **training_kwargs)
    training_time = time.time() - tic

    tokenizer.save_pretrained(args.output)

    # EVALUATION

    if args.verbose:
        print("Evaluate tokenizer")

    info.update(
        {
            "training_time": training_time,
            "vocab_size": tokenizer.vocab_size,
        }
    )

    info.update(
        {
            "example_after": {
                example_sentence: test_tokenizer(tokenizer, example_sentence)
            }
        }
    )

    json.dump(
        info,
        open(f"{args.output}/training_info.json", "w", encoding="utf8"),
        indent=2,
        ensure_ascii=False,
    )

    for train in (
        True,
        False,
    ):
        all_byte_tokens = [
            i
            for i, t in enumerate(
                tokenizer.convert_ids_to_tokens(range(tokenizer.vocab_size))
            )
            if re.match(r"<0x.*>$", t)
        ]
        total_num_pages = 0
        total_num_paragraph = 0
        total_num_lines = 0
        total_num_words = 0
        total_num_chars = 0
        total_num_bytes = 0
        total_num_tokens = 0
        total_num_tokens_single_byte = 0
        tic = time.time()
        _, dataset = tokenizer_dataset(train=train, streaming=True, debug=args.debug)
        for text in dataset:
            total_num_pages += 1
            total_num_paragraph += len(text.split("\n\n"))
            total_num_lines += len(text.split("\n"))
            total_num_words += len(text.split())
            total_num_chars += len(text)
            total_num_bytes += len(bytes(text, "utf-8"))
            tokens = tokenizer.encode(text)
            byte_tokens = [t for t in tokens if t in all_byte_tokens]
            total_num_tokens += len(tokens)
            total_num_tokens_single_byte += len(byte_tokens)
        toc = time.time()

        subset = "train" if train else "eval"
        info.update(
            {
                f"{subset}_tokenization_time": toc - tic,
                f"{subset}_num_pages": total_num_pages,
                f"{subset}_num_paragraph": total_num_paragraph,
                f"{subset}_num_lines": total_num_lines,
                f"{subset}_num_words": total_num_words,
                f"{subset}_num_chars": total_num_chars,
                f"{subset}_num_bytes": total_num_bytes,
                f"{subset}_num_tokens": total_num_tokens,
                f"{subset}_num_tokens_single_byte": total_num_tokens_single_byte,
                f"{subset}_avg_length_token_char": total_num_chars / total_num_tokens,
                f"{subset}_avg_length_token_byte": total_num_bytes / total_num_tokens,
                f"{subset}_avg_byte_kept": total_num_tokens_single_byte
                / total_num_bytes,
            }
        )

        json.dump(
            info,
            open(f"{args.output}/training_info.json", "w", encoding="utf8"),
            indent=2,
            ensure_ascii=False,
        )
