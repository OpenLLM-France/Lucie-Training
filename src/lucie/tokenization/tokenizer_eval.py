# https://github.com/huggingface/transformers/pull/22264
# Byte Fallback Tokenizer

# https://github.com/huggingface/tokenizers/issues/1407#issue-2028070675

import re

import pandas as pd
import transformers
from tokenizer_train import set_infinite_length

from data import decompose_datasets, tokenizer_dataset

if __name__ == "__main__":
    import argparse
    import os
    import time

    import tqdm

    parser = argparse.ArgumentParser(description="Evaluate a tokenizer.")
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
        "tokenizer",
        help="Tokenizer to evaluate",
    )
    parser.add_argument(
        "--regex",
        default=None,
        type=str,
        help="only evaluate datasets matching this regex",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory",
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="Batch size",
    )
    args = parser.parse_args()

    if args.tokenizer.lower() in ["gpt-4"]:
        import tiktoken

        tokenizer = tiktoken.encoding_for_model("gpt-4")
        all_byte_tokens = []

    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        tokenizer = set_infinite_length(tokenizer)

        all_byte_tokens = [
            i
            for i, t in enumerate(tokenizer.convert_ids_to_tokens(range(tokenizer.vocab_size)))
            if re.match(r"<0x.*>$", t)
        ]

        if not all_byte_tokens:
            offset = len(tokenizer.all_special_tokens)
            all_byte_tokens = list(range(offset, offset + 256))

        if not os.path.exists(args.tokenizer):
            os.makedirs(args.tokenizer, exist_ok=True)
            tokenizer.save_pretrained(args.tokenizer)

    if args.output is None:
        args.output = args.tokenizer
    os.makedirs(args.output, exist_ok=True)

    output_file = f"{args.output}/eval_transformers_2.csv"

    already_computed = []
    eval_data = []
    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
        eval_data = df.values.tolist()
        already_computed = [d[0] for d in eval_data]

    # EVALUATION
    for dataset in tqdm.tqdm(
        list(decompose_datasets(tokenizer_dataset(train=False, factor=1))), desc="Evaluating datasets"
    ):
        name = dataset.name
        if args.regex is not None and not re.match(re.escape(args.regex), name, re.IGNORECASE):
            print(f"Skipping eval of {args.tokenizer} on {name} (regex mismatch)")
            continue
        if name in already_computed:
            print(f"Skipping eval of {args.tokenizer} on {name} (already computed)")
            continue
        print(f"Evaluate {args.tokenizer} on {name}...")

        total_num_pages = 0
        total_num_paragraph = 0
        total_num_lines = 0
        total_num_words = 0
        total_num_chars = 0
        total_num_bytes = 0
        total_num_tokens = 0
        total_num_spaces = 0
        total_num_linebreaks = 0
        total_num_tabs = 0
        total_num_digits = 0
        total_num_tokens_space = 0
        total_num_tokens_linebreak = 0
        total_num_tokens_tab = 0
        total_num_tokens_digit = 0
        total_num_tokens_single_byte = 0

        def update_stats(text, tokens):
            global tokenizer
            global total_num_pages
            global total_num_paragraph
            global total_num_lines
            global total_num_words
            global total_num_chars
            global total_num_bytes
            global total_num_tokens
            global total_num_spaces
            global total_num_linebreaks
            global total_num_tabs
            global total_num_digits
            global total_num_tokens_space
            global total_num_tokens_linebreak
            global total_num_tokens_tab
            global total_num_tokens_digit
            global total_num_tokens_single_byte
            total_num_pages += 1
            total_num_paragraph += len(text.split("\n\n"))
            total_num_lines += len(text.split("\n"))
            total_num_words += len(text.split())
            total_num_chars += len(text)
            total_num_bytes += len(bytes(text, "utf-8"))
            total_num_spaces += text.count(" ")
            total_num_linebreaks += text.count("\n")
            total_num_tabs += text.count("\t")
            total_num_digits += sum(c.isdigit() for c in text)
            total_num_tokens += len(tokens)
            if hasattr(tokenizer, "convert_ids_to_tokens"):
                token_str = tokenizer.convert_ids_to_tokens(tokens)
            else:
                token_str = tokenizer.decode_batch([[t] for t in tokens])
            total_num_tokens_space += sum(not t.strip(" ▁") for t in token_str)
            total_num_tokens_linebreak += sum(not t.strip("\n") for t in token_str)
            total_num_tokens_tab += sum(not t.strip("\t") for t in token_str)
            total_num_tokens_digit += sum(not re.sub(r"[0-9]", "", t) for t in token_str)
            total_num_tokens_single_byte += sum(t in all_byte_tokens for t in tokens)

        use_batch = args.batch_size > 1
        update_each = args.batch_size if use_batch else 32

        def process_batch(batch, use_batch=use_batch):
            global tokenizer
            global processing_time
            tic = time.time()
            if use_batch:
                tokens = tokenizer.batch_encode_plus(batch).input_ids
            else:
                tokens = [tokenizer.encode(t) for t in batch]
            processing_time += time.time() - tic
            assert len(tokens) == len(batch)
            for t, b in zip(tokens, batch):
                update_stats(b, t)

        processing_time = 0
        batch = []
        for text in tqdm.tqdm(dataset, total=len(dataset), desc=f"Evaluating {name}"):
            batch.append(text)
            if len(batch) == update_each:
                process_batch(batch)
                batch = []

        if len(batch):
            process_batch(batch)

        print(f"Adding {name} in {output_file}...")
        if os.path.exists(output_file):
            df = pd.read_csv(output_file)
            eval_data = df.values.tolist()
            already_computed = [d[0] for d in eval_data]

        eval_data.append(
            [
                name,
                total_num_pages,
                total_num_paragraph,
                total_num_lines,
                total_num_words,
                total_num_chars,
                total_num_bytes,
                total_num_spaces,
                total_num_linebreaks,
                total_num_tabs,
                total_num_digits,
                total_num_tokens,
                total_num_tokens_space,
                total_num_tokens_linebreak,
                total_num_tokens_tab,
                total_num_tokens_digit,
                total_num_tokens_single_byte,
                args.batch_size,
                processing_time,
            ]
        )

        df = pd.DataFrame(
            eval_data,
            columns=[
                "name",
                "num_pages",
                "num_paragraph",
                "num_lines",
                "num_words",
                "num_chars",
                "num_bytes",
                "num_spaces",
                "num_linebreaks",
                "num_tabs",
                "num_digits",
                "num_tokens",
                "num_tokens_space",
                "num_tokens_linebreak",
                "num_tokens_tab",
                "num_tokens_digit",
                "num_tokens_single_byte",
                "batch_size",
                "processing_time",
            ],
        )

        df.to_csv(output_file, index=False)
