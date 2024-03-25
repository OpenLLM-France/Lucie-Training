# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Processing large data for pretraining."""
import argparse
import json
import multiprocessing
import os
import sys
import time

from data import decompose_datasets, get_datasets

nltk_available = False
# try:
#     import nltk
#     nltk_available = True
# except ImportError:
#     nltk_available = False

megatron_deepspeed_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "Megatron-DeepSpeed"))
sys.path = [megatron_deepspeed_folder] + sys.path  # Better to prepend for "tools" module

from megatron.data import indexed_dataset  # noqa # E402 Module level import not at top of file
from megatron.tokenizer import build_tokenizer  # noqa # E402 Module level import not at top of file


class IdentitySplitter:
    def tokenize(self, *text):
        return text


class Encoder:
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = build_tokenizer(self.args)
        if False:  # self.args.split_sentences:
            pass
            # if not nltk_available:
            #     print("NLTK is not available to split sentences.")
            #     exit()
            # library = f"tokenizers/punkt/{self.args.lang}.pickle"
            # splitter = nltk.load(library)
            # if self.args.keep_newlines:
            #     # this prevents punkt from eating newlines after sentences
            #     Encoder.splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
            #         train_text=splitter._params, lang_vars=CustomLanguageVars()
            #     )
            # else:
            #     Encoder.splitter = splitter
        else:
            Encoder.splitter = IdentitySplitter()

    def split(self, json_line):
        data = json.loads(json_line)
        output = {}
        for key in self.args.json_keys:
            text = data[key]
            max_len = 1000000
            tokens_list = [Encoder.splitter.tokenize(text[i : i + max_len]) for i in range(0, len(text), max_len)]
            output[key] = [tokens for partial in tokens_list for tokens in partial]
        return json.dumps(output), len(json_line)

    def encode(self, json_line):
        data = json.loads(json_line)
        ids = {}
        lens = {}
        for key in self.args.json_keys:
            text = data[key]
            if isinstance(text, list):
                sentences = text
            else:
                sentences = [text]
            doc_ids = []
            sentence_lens = []
            for sentence in sentences:
                # JL: remove trailing whitespaces and line breaks
                sentence = sentence.rstrip()
                sentence_ids = Encoder.tokenizer.tokenize(sentence)
                if len(sentence_ids) > 0:
                    doc_ids.extend(sentence_ids)
                    sentence_lens.append(len(sentence_ids))
            if len(doc_ids) > 0 and self.args.append_eod:
                doc_ids.append(Encoder.tokenizer.eod)
            ids[key] = doc_ids
            lens[key] = sentence_lens
        return ids, lens, len(json_line)


class Partition:
    def __init__(self, args, workers):
        self.args = args
        self.workers = workers

    def print_processing_stats(self, count, proc_start, total_bytes_processed):
        current = time.time()
        elapsed = current - proc_start
        mbs = total_bytes_processed / elapsed / 1024 / 1024
        print(f"Processed {count} documents", f"({count/elapsed} docs/s, {mbs} MB/s).", file=sys.stderr)
        sys.stdout.flush()

    def split_sentences(self, file_name):
        input_file_name, output_file_name = file_name
        print("Opening", input_file_name)
        fin = open(input_file_name, encoding="utf-8")
        fout = open(output_file_name, "w")

        encoder = Encoder(self.args)
        pool = multiprocessing.Pool(self.args.threads_tokenization, initializer=encoder.initializer)
        split_docs = pool.imap(encoder.split, fin, 32)

        proc_start = time.time()
        total_bytes_processed = 0
        for i, (doc, bytes_processed) in enumerate(split_docs, start=1):
            total_bytes_processed += bytes_processed
            fout.write(doc + "\n")
            if i % self.args.log_interval == 0:
                self.print_processing_stats(i, proc_start, total_bytes_processed)

        fin.close()
        fout.close()

    def process_json_file(self, input_file_name, output_prefix):
        # print("Opening", input_file_name)
        fin = open(input_file_name, encoding="utf-8")

        # startup_start = time.time()
        encoder = Encoder(self.args)
        tokenizer = build_tokenizer(self.args)
        if self.args.threads_tokenization > 1:
            raise NotImplementedError("Multithreading is not supported for encoding.")
            # pool = multiprocessing.Pool(self.args.threads_tokenization, initializer=encoder.initializer)
            # encoded_docs = pool.imap(encoder.encode, fin, 32)
        else:
            encoder.initializer()
            encoded_docs = map(encoder.encode, fin)

        level = "document"
        # if self.args.split_sentences:
        #     level = "sentence"

        output_bin_files = {}
        output_idx_files = {}
        builders = {}

        for key in self.args.json_keys:
            output_bin_files[key] = f"{output_prefix}_{key}_{level}.bin"
            output_idx_files[key] = f"{output_prefix}_{key}_{level}.idx"
            builders[key] = indexed_dataset.make_builder(
                output_bin_files[key], impl=self.args.dataset_impl, vocab_size=tokenizer.vocab_size
            )

        # startup_end = time.time()
        proc_start = time.time()
        total_bytes_processed = 0
        # print("Time to startup:", startup_end - startup_start)
        i = -1
        for i, (doc, sentence_lens, bytes_processed) in enumerate(encoded_docs, start=1):
            total_bytes_processed += bytes_processed
            for key in doc.keys():
                builders[key].add_doc(doc[key], sentence_lens[key])
            if i % self.args.log_interval == 0:
                pass
                # self.print_processing_stats(i, proc_start, total_bytes_processed)
        assert i >= 0, f"Error: {input_file_name} is empty."
        self.print_processing_stats(i, proc_start, total_bytes_processed)

        fin.close()
        builders[key].finalize(output_idx_files[key])

    def process_dataset(self, dataset_name):  # noqa # C901 `...` is too complex
        global error_flag, num_processes

        remove_jsonl = self.args.remove_jsonl
        has_increased_num_processes = False

        try:
            global all_datas
            dataset = all_datas[dataset_name]

            expected_file = os.path.join(self.args.output_folder, dataset_name + "_text_document.idx")
            if isinstance(dataset, str):
                jsonl_file = dataset
                assert os.path.exists(jsonl_file), f"Error: {jsonl_file} does not exist."
                remove_jsonl = False
            else:
                jsonl_file = os.path.join(self.args.jsonl_folder, dataset_name + ".jsonl")

            if os.path.exists(expected_file) and self.args.remove_jsonl:
                # print(f"Skipping {jsonl_file} as {expected_file} exists.")
                sys.stdout.flush()
                return

            if not os.path.exists(jsonl_file):
                print(f"Writing {jsonl_file}...")
                os.makedirs(os.path.dirname(jsonl_file), exist_ok=True)
                tic = time.time()
                try:
                    with open(jsonl_file, "w", encoding="utf-8") as f:
                        for doc in dataset:
                            f.write(json.dumps({self.args.json_keys[0]: doc}) + "\n")
                except (Exception, KeyboardInterrupt) as err:
                    if os.path.exists(jsonl_file):
                        os.remove(jsonl_file)
                    raise err
                print(f"Wrote {jsonl_file} in {time.time() - tic:.2f}s")
                sys.stdout.flush()

            all_datas.pop(dataset_name)
            del dataset

            if os.path.exists(expected_file):
                # print(f"Skipping {jsonl_file} as {expected_file} exists.")
                sys.stdout.flush()
                return

            num_processes.value += 1
            has_increased_num_processes = True

            print(f"Processing {jsonl_file} ({num_processes.value} processes)")
            sys.stdout.flush()

            os.makedirs(self.args.output_folder, exist_ok=True)
            output_prefix = os.path.join(self.args.output_folder, dataset_name)

            self.process_json_file(jsonl_file, output_prefix)
            print(f"Processed {jsonl_file}...")
            sys.stdout.flush()

            if remove_jsonl:
                os.remove(jsonl_file)

        except (Exception, KeyboardInterrupt) as err:
            print(f"Error processing {dataset_name}: {err}")
            sys.stdout.flush()
            if error_flag is not None:
                error_flag.value = True
            if has_increased_num_processes:
                num_processes.value -= 1

        if has_increased_num_processes:
            num_processes.value -= 1


global error_flag
error_flag = multiprocessing.Value("b", False)
global num_processes
num_processes = multiprocessing.Value("i", 0)


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="input data")
    group.add_argument("--datasets", type=str, default="all", help="Datasets")
    group.add_argument(
        "--json-keys", nargs="+", default=["text"], help="space separate listed of keys to extract from json"
    )
    # group.add_argument("--split-sentences", action="store_true", help="Split documents into sentences.")
    group.add_argument("--keep-newlines", action="store_true", help="Keep newlines between sentences when splitting.")

    group = parser.add_argument_group(title="tokenizer")
    group.add_argument(
        "--tokenizer-type",
        type=str,
        default="PretrainedFromHF",
        choices=[
            "BertWordPieceLowerCase",
            "BertWordPieceCase",
            "GPT2BPETokenizer",
            "SentencePieceTokenizer",
            "GPTSentencePieceTokenizer",
            "PretrainedFromHF",
            "NullTokenizer",
        ],
        help="What type of tokenizer to use.",
    )
    group.add_argument("--tokenizer-model", type=str, default=None, help="Sentencepiece tokenizer model.")
    group.add_argument(
        "--tokenizer-name-or-path",
        type=str,
        default="OpenLLM-France/Lucie-tokenizer-v2.4-space_prefix_all",
        help="Pretrained tokenizer name or path",
    )
    group.add_argument("--vocab-file", type=str, default=None, help="Path to the vocab file")
    group.add_argument("--vocab-size", default=786, help="size of vocab for use with NullTokenizer")
    group.add_argument("--merge-file", type=str, default=None, help="Path to the BPE merge file (if necessary).")
    group.add_argument("--append-eod", action="store_true", help="Append an <eod> token to the end of a document.")
    group.add_argument(
        "--lang", type=str, default="french", help="Language to use for NLTK-powered sentence splitting."
    )
    group = parser.add_argument_group(title="output data")
    group.add_argument("--output-folder", type=str, default="tokenized_data", help="Output folder")
    group.add_argument("--jsonl-folder", type=str, default="tmp_to_tokenize", help="Folder with jsonl files")
    group.add_argument("--dataset-impl", type=str, default="mmap", choices=["lazy", "cached", "mmap"])

    group = parser.add_argument_group(title="runtime")
    group.add_argument(
        "--workers",
        type=int,
        default=9,
        help=(
            "Number of worker processes to launch."
            "A good default for fast pre-processing "
            "is: (workers * partitions) = available CPU cores."
        ),
    )
    group.add_argument(
        "--threads_tokenization", type=int, default=1, help="Number of sub-threads for tokenizing each subset part"
    )
    group.add_argument("--partitions", type=int, default=1, help="Number of file partitions")
    group.add_argument("--log-interval", type=int, default=1000, help="Interval between progress updates")
    args = parser.parse_args()

    args.keep_empty = False

    # some default/dummy values for the tokenizer
    args.rank = 1
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    return args


def get_file_name(args, file_id):
    file_name, extension = os.path.splitext(args.input)
    input_file_name = file_name + "_" + str(file_id) + extension
    sentence_split_file = file_name + "_ss_" + str(file_id) + extension
    output_prefix = args.output_prefix + "_" + str(file_id)
    file_names = {"partition": input_file_name, "sentence_split": sentence_split_file, "output_prefix": output_prefix}
    return file_names


def check_files_exist(in_ss_out_names, key, num_partitions):
    for i in range(num_partitions):
        if not os.path.exists(in_ss_out_names[i][key]):
            return False
    return True


def dataset_to_key_value(dataset):
    if isinstance(dataset, tuple) and len(dataset) == 2:
        return dataset
    else:
        return (dataset.name.replace(":", "--"), dataset)


def main():
    args = get_args()

    args.remove_jsonl = "tmp" in args.jsonl_folder

    global all_datas
    all_datas = get_datasets(args.datasets)
    all_datas = dict(
        dataset_to_key_value(dataset)
        for dataset in decompose_datasets(all_datas, parquet_level=True, return_json_file_if_possible=True)
    )

    partition = Partition(args, args.workers)

    print("=" * 20)
    print(f"Processing {len(all_datas)} datasets with {args.workers} workers.")

    # # Shared flag to indicate if an error occurred in any process
    # error_flag = multiprocessing.Value('b', False)

    with multiprocessing.Pool(processes=args.workers) as pool:
        # Partially apply the process function with error_flag argument
        # import functools
        # process_dataset = functools.partial(partition.process_dataset, error_flag=error_flag)

        chunk_size = 1  # args.workers # len(all_datas) // args.workers
        for _ in pool.imap_unordered(partition.process_dataset, sorted(all_datas.keys()), chunk_size):
            # Check if any error occurred
            if error_flag.value:
                # If an error occurred, terminate all processes in the pool
                print("An error occurred in one of the processes. Terminating all processes.")
                sys.stdout.flush()
                pool.terminate()
                break

        # pool.close()
        # pool.join()


if __name__ == "__main__":
    main()
