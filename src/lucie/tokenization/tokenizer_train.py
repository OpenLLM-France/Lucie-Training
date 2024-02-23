# https://github.com/huggingface/transformers/pull/22264
# Byte Fallback Tokenizer

# https://github.com/huggingface/tokenizers/issues/1407#issue-2028070675

import sys
import tempfile

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
        name = "gpt2"  # same as "allenai/OLMo-1B" without <|padding|> token
        # name = "meta-llama/Llama-2-7b-hf"
        base_tokenizer = transformers.AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        base_tokenizer.add_special_tokens({"bos_token": "<|startoftext|>"})

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

    base_tokenizer = set_infinite_length(base_tokenizer)

    return name.replace("/", "--"), base_tokenizer


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
        "--no_persee",
        dest="persee",
        action="store_false",
        default=True,
        help="Don't use Persee",
    )
    parser.add_argument(
        "--no_legi",
        dest="legi",
        action="store_false",
        default=True,
        help="Don't use LEGI",
    )
    parser.add_argument(
        "--no_europarl",
        dest="europarl",
        action="store_false",
        default=True,
        help="Don't use Europarl",
    )
    parser.add_argument(
        "--use_sentence_piece",
        default=False,
        action="store_true",
        help="To use sentence piece (spm_train)",
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

    if args.use_sentence_piece:
        name_tokenizer = f"smp{'_bytefallback' if args.byte_fallback else ''}_{args.vocab_size}"
        base_tokenizer = None
        eos_piece = "</s>"
    else:
        name_tokenizer, base_tokenizer = make_tokenizer(args.byte_fallback)
        name_tokenizer += f"_{args.vocab_size}"
        eos_piece = base_tokenizer.eos_token

    example_sentence = (
        f"   [INST] Coucou [/INST] Hello {eos_piece} [INST] "
        "Mais en Français, comment est-ce que ça se passera ? 1999, 2000, 2002.2, 1 000 000, 1\u00A0000\u00A0000"
        "[/INST] Eh bien ␣ alors あ゙"
    )

    info = {}
    if base_tokenizer is not None:
        info.update({"example_before": {example_sentence: test_tokenizer(base_tokenizer, example_sentence)}})
        print(json.dumps(info, indent=2, ensure_ascii=False))

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
    if args.use_sentence_piece:
        convert_script = "thirdparty_tokenizers/bindings/python/scripts/sentencepiece_extractor.py"
        assert os.path.isfile(convert_script), f"File {convert_script} does not exist"

        with tempfile.NamedTemporaryFile(suffix=".txt") as f:
            print(f"Writing to {f.name}")

            cmd_line = f"""\
spm_train \
    {f.name} \
    --model_prefix "{args.output}/tokenizer" \
    --input tmp.txt \
    --input_format text \
    --model_type bpe \
    --vocab_size {args.vocab_size} \
    --split_digits true \
    --allow_whitespace_only_pieces true \
    --byte_fallback false \
    --max_sentence_length 10000000 \
"""
            for text in trainset:
                f.write((text + "\n").encode("utf-8"))
            f.flush()
            print(cmd_line)
            code = os.system(cmd_line)
            assert code == 0, f"Error code {code} for command {cmd_line}"
            vocab_file = f"{args.output}/tokenizer.vocab"
            model_file = f"{args.output}/tokenizer.model"
            assert os.path.exists(vocab_file), f"File {vocab_file} does not exist"
            assert os.path.exists(model_file), f"File {model_file} does not exist"

            vocab_file2 = f"{args.output}/tokenizer_vocab.json"
            merge_file = f"{args.output}/merges.txt"

            cmd_line = f"""{sys.executable} {convert_script} \
                --provider sentencepiece --model {model_file} \
                --vocab-output-path {vocab_file2} \
                --merges-output-path {merge_file}"""
            print(cmd_line)
            code = os.system(cmd_line)
            assert code == 0, f"Error code {code} for command {cmd_line}"

            tokenizer = tokenizers.ByteLevelBPETokenizer.from_file(vocab_file2, merge_file)
            tokenizer.save_model(args.output)

            from transformers import PreTrainedTokenizerFast

            tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
            tokenizer.save_pretrained(args.output)

            # from transformers.convert_slow_tokenizer import GPT2Converter
            # converted = GPT2Converter(tokenizer).converted()

    else:
        training_kwargs = {
            "vocab_size": args.vocab_size,
            "min_frequency": 0,
            "show_progress": True,
            # "max_token_length": 10**19,
            # initial_alphabet=["▁[", "]"],
        }

        print("Adding individual digits preproc")

        def set_individual_digits(tokenizer):
            tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Sequence(
                [
                    tokenizers.pre_tokenizers.Digits(individual_digits=True),
                    tokenizer.pre_tokenizer,
                ]
            )
            return tokenizer

        training_kwargs["configure_tokenizer"] = set_individual_digits

        tokenizer = base_tokenizer.train_new_from_iterator(trainset, **training_kwargs)
        tokenizer = set_infinite_length(tokenizer)
        tokenizer.save_pretrained(args.output)

    training_time = time.time() - tic

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.output)

    # EVALUATION

    if args.verbose:
        print("Evaluate tokenizer")

    info.update(
        {
            "training_time": training_time,
            "vocab_size": tokenizer.vocab_size,
        }
    )

    info.update({"example_after": {example_sentence: test_tokenizer(tokenizer, example_sentence)}})

    json.dump(
        info,
        open(f"{args.output}/training_info.json", "w", encoding="utf8"),
        indent=2,
        ensure_ascii=False,
    )
