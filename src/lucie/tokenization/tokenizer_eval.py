# https://github.com/huggingface/transformers/pull/22264
# Byte Fallback Tokenizer

# https://github.com/huggingface/tokenizers/issues/1407#issue-2028070675

import re

import pandas as pd
import tiktoken
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
        tokenizer = tiktoken.encoding_for_model("gpt-4")
        all_byte_tokens = []

    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
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

    output_file = f"{args.output}/eval_transformers_{args.batch_size}.csv"

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
        total_num_tokens_single_byte = 0
        tic = time.time()
        batch = []
        for text in tqdm.tqdm(dataset, total=len(dataset), desc=f"Evaluating {name}"):
            if args.batch_size == 1:
                total_num_pages += 1
                total_num_paragraph += len(text.split("\n\n"))
                total_num_lines += len(text.split("\n"))
                total_num_words += len(text.split())
                total_num_chars += len(text)
                total_num_bytes += len(bytes(text, "utf-8"))
                tokens = tokenizer.encode(text)
                total_num_tokens += len(tokens)
                byte_tokens = [t for t in tokens if t in all_byte_tokens]
                total_num_tokens_single_byte += len(byte_tokens)
            else:
                batch.append(text)
                if len(batch) == args.batch_size:
                    total_num_pages += len(batch)
                    total_num_paragraph += sum(len(text.split("\n\n")) for text in batch)
                    total_num_lines += sum(len(text.split("\n")) for text in batch)
                    total_num_words += sum(len(text.split()) for text in batch)
                    total_num_chars += sum(len(text) for text in batch)
                    total_num_bytes += sum(len(bytes(text, "utf-8")) for text in batch)
                    tokens = tokenizer.batch_encode_plus(batch).input_ids
                    total_num_tokens += sum(len(t) for t in tokens)
                    byte_tokens = [[t for t in tok if t in all_byte_tokens] for tok in tokens]
                    total_num_tokens_single_byte += sum(len(t) for t in byte_tokens)
                    batch = []
        toc = time.time()

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
                total_num_tokens,
                total_num_tokens_single_byte,
                total_num_chars / total_num_tokens,
                total_num_bytes / total_num_tokens,
                total_num_tokens_single_byte / total_num_bytes,
                toc - tic,
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
                "num_tokens",
                "num_tokens_single_byte",
                "avg_length_token_char",
                "avg_length_token_byte",
                "avg_byte_kept",
                "tokenization_time",
            ],
        )

        print(df)
        df.to_csv(output_file, index=False)
