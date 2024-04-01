"""Processing large data for pretraining."""
import argparse
import json
import multiprocessing
import os
import random
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

    def encode(self, text, key=None):
        if key is None:
            key = self.args.json_keys[0]
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
        return {key: doc_ids}, {key: sentence_lens}, len(text)

    def encode_json(self, json_line):
        data = json.loads(json_line)
        ids = {}
        lens = {}
        for key in self.args.json_keys:
            text = data[key]
            _id, _len, _ = self.encode(text, key=key)
            ids.update(_id)
            lens.update(_len)
        return ids, lens, len(json_line)


def current_date():
    return time.strftime("%Y-%m-%d %H:%M:%S")


class Partition:
    def __init__(self, args, workers):
        self.args = args
        self.workers = workers
        self.encoder = None
        tokenizer = build_tokenizer(self.args)
        self.vocab_size = tokenizer.vocab_size

    def initializer(self):
        self.encoder = Encoder(self.args)
        self.encoder.initializer()

    def print_processing_stats(self, count, proc_start, total_bytes_processed, dataset_name):
        current = time.time()
        elapsed = current - proc_start
        mbs = total_bytes_processed / elapsed / 1024 / 1024
        print(
            f"{current_date()} -- {dataset_name}: Processed {count} documents in {elapsed} seconds",
            f"({count/elapsed} docs/s, {mbs} MB/s).",
            file=sys.stderr,
        )
        sys.stdout.flush()

    def process_batch(self, input, output_prefix):  # noqa # C901 `...` is too complex
        """
        Process a single json file or a list of text

        Args:
            input: str or list of str
                input json file or list of text
            output_prefix: str
                prefix for output files
        """

        # startup_start = time.time()
        if self.encoder is None:
            self.initializer()
        encoder = self.encoder

        dataset_name = os.path.basename(output_prefix)

        if isinstance(input, str):
            # print("Opening", input)
            fin = open(input, encoding="utf-8")
            encoded_docs = map(encoder.encode_json, fin)
        else:
            fin = None
            encoded_docs = map(encoder.encode, input)

        level = "document"

        output_bin_files = {}
        output_idx_files = {}
        builders = {}

        for key in self.args.json_keys:
            output_bin_files[key] = f"{output_prefix}_{key}_{level}.bin"
            output_idx_files[key] = f"{output_prefix}_{key}_{level}.idx"
            builders[key] = indexed_dataset.make_builder(
                output_bin_files[key], impl=self.args.dataset_impl, vocab_size=self.vocab_size
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
                self.print_processing_stats(i, proc_start, total_bytes_processed, dataset_name)

        # Do not fail on empty dataset (can happen when filtering documents)
        if i < 0:
            print(f"WARNING: {input} is empty.")
            sys.stdout.flush()
            for key in self.args.json_keys:
                filename = output_bin_files[key]
                if os.path.exists(filename):
                    os.remove(filename)
            return

        assert i >= 0, f"Error: {input} is empty."
        self.print_processing_stats(i, proc_start, total_bytes_processed, dataset_name)

        builders[key].finalize(output_idx_files[key])

        if fin is not None:
            fin.close()

    def process_dataset(self, dataset_name, use_jsonl_file=False):  # noqa # C901 `...` is too complex
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
                if use_jsonl_file:
                    jsonl_file = os.path.join(self.args.jsonl_folder, dataset_name + ".jsonl")
                else:
                    jsonl_file = None

            if os.path.exists(expected_file) and self.args.remove_jsonl:
                # print(f"Skipping {jsonl_file} as {expected_file} exists.")
                sys.stdout.flush()
                return

            if jsonl_file and not os.path.exists(jsonl_file):
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

            if os.path.exists(expected_file):
                # print(f"Skipping {jsonl_file} as {expected_file} exists.")
                sys.stdout.flush()
                return

            num_processes.value += 1
            has_increased_num_processes = True

            os.makedirs(self.args.output_folder, exist_ok=True)
            output_prefix = os.path.join(self.args.output_folder, dataset_name)

            if jsonl_file:
                del dataset

                print(
                    f"{current_date()} -- Processing {jsonl_file} -> {dataset_name} ({num_processes.value} processes)"
                )
                sys.stdout.flush()

                self.process_batch(jsonl_file, output_prefix)
                print(f"{current_date()} -- Processed {jsonl_file}...")
                sys.stdout.flush()

                if remove_jsonl:
                    os.remove(jsonl_file)

            else:
                print(f"{current_date()} -- Processing {dataset_name} ({num_processes.value} processes)")
                sys.stdout.flush()

                self.process_batch(dataset, output_prefix)
                print(f"{current_date()} -- Processed {dataset_name}...")
                sys.stdout.flush()
                del dataset

        except (Exception, KeyboardInterrupt) as err:
            import traceback

            print(traceback.format_exc())
            print(f"{current_date()} -- Error processing {dataset_name}: {err}")
            sys.stdout.flush()
            if error_flag is not None:
                error_flag.value = True
            if has_increased_num_processes:
                num_processes.value -= 1
                has_increased_num_processes = False

        if has_increased_num_processes:
            num_processes.value -= 1
            has_increased_num_processes = False


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
        default="OpenLLM-France/Lucie-tokenizer-v2.9",
        help="Pretrained tokenizer name or path",
    )
    group.add_argument("--vocab-file", type=str, default=None, help="Path to the vocab file")
    group.add_argument("--vocab-size", default=786, help="size of vocab for use with NullTokenizer")
    group.add_argument("--merge-file", type=str, default=None, help="Path to the BPE merge file (if necessary).")
    group.add_argument("--append-eod", action="store_true", help="Append an <eod> token to the end of a document.")
    group = parser.add_argument_group(title="output data")
    group.add_argument("--output-folder", type=str, default="tokenized_data", help="Output folder")
    group.add_argument("--jsonl-folder", type=str, default="tmp_to_tokenize", help="Folder with jsonl files")
    group.add_argument("--dataset-impl", type=str, default="mmap", choices=["lazy", "cached", "mmap"])

    group = parser.add_argument_group(title="runtime")
    group.add_argument(
        "--workers",
        type=int,
        default=10,
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
    input = file_name + "_" + str(file_id) + extension
    sentence_split_file = file_name + "_ss_" + str(file_id) + extension
    output_prefix = args.output_prefix + "_" + str(file_id)
    file_names = {"partition": input, "sentence_split": sentence_split_file, "output_prefix": output_prefix}
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

    print(f"Output folder: {args.output_folder}")
    tokenizer_folder = os.path.join(args.output_folder, "tokenizer")
    if True:  # not os.path.exists(tokenizer_folder):
        os.makedirs(tokenizer_folder, exist_ok=True)
        import transformers

        tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
        tokenizer.save_pretrained(tokenizer_folder)

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

    all_data_names = list(all_datas.keys())
    # Suffle the data names to avoid processing the largest datasets first
    random.shuffle(all_data_names)

    with multiprocessing.Pool(
        processes=args.workers, initializer=partition.initializer
    ) as pool:  # , maxtasksperchild=1
        # Partially apply the process function with error_flag argument
        # import functools
        # process_dataset = functools.partial(partition.process_dataset, error_flag=error_flag)

        chunk_size = len(all_datas) // args.workers
        for _ in pool.imap_unordered(partition.process_dataset, all_data_names, chunk_size):
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
