import json
import os
import sys

import tqdm

rootdir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
megatron_deepspeed_folder = os.path.join(rootdir, "Megatron-DeepSpeed")
sys.path = [megatron_deepspeed_folder] + sys.path  # Better to prepend for "tools" module
from megatron.data import indexed_dataset  # noqa # E402 Module level import not at top of file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Split megatron MMap Indexed dataset(s) into datasets with short and long documents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("inputs", type=str, nargs="+", help="Input indexed dataset filenames (without extension)")
    parser.add_argument("output", type=str, help="output folder")
    parser.add_argument(
        "--vocab_size", type=int, default=65024, help="Vocabulary size (is it larger or not than 65500?)"
    )
    parser.add_argument(
        "--tokens_split", type=int, default=4096, help="Token size threshold (to decide if a document is short or long)"
    )
    parser.add_argument(
        "--collect-stats", "-n", action="store_true", default=False, help="Only collect stats about document lengths"
    )
    args = parser.parse_args()

    vocab_size = args.vocab_size

    def get_filenames():
        for f in args.inputs:
            if os.path.isdir(f):
                for root, _, files in os.walk(f):
                    for file in files:
                        if file.endswith(".idx"):
                            yield os.path.join(root, os.path.splitext(file)[0])
            elif os.path.isfile(f):
                yield os.path.splitext(f)[0]
            else:
                raise ValueError(f"Invalid input: {f} (must be a file or a folder)")

    max_len_for_stats = 10000 if args.collect_stats else None

    all_paths = list(get_filenames())

    # Sort files by increasing size
    all_paths.sort(key=lambda x: os.path.getsize(x + ".bin"))

    # Ugly hack to filter-out web snapshots that are not the last one
    def select_file(path):
        if "RedPajama" in path:
            return "2023" in path
        if "FineWebEdu" in path:
            return "2024" in path
        return True

    all_paths = [path for path in all_paths if select_file(path)]

    all_paths = [
        path
        for path in all_paths
        if (
            not os.path.exists(
                os.path.join(args.output, f"length-0-{args.tokens_split}", os.path.basename(path) + ".bin")
            )
            and not os.path.exists(
                os.path.join(args.output, f"length-{args.tokens_split}-inf", os.path.basename(path) + ".bin")
            )
        )
    ]

    assert len(all_paths) > 0, "No files to process"

    progress_bar = tqdm.tqdm(all_paths, desc="Splitting datasets")  # , dynamic_ncols=True)
    for path in progress_bar:
        progress_bar.set_description(f"Processing {os.path.basename(path)}")

        _lengths = []
        stats_small = {"total_tokens": 0, "total_sequences": 0, "min_tokens": 1e10, "max_tokens": 0}
        stats_large = stats_small.copy()

        # Aggregate data and write output bin

        if args.collect_stats:
            output_length_stats = os.path.join("length_stats", os.path.basename(path) + ".json")
            if os.path.exists(output_length_stats):
                continue

        else:
            output_tokens_small = os.path.join(args.output, f"length-0-{args.tokens_split}", os.path.basename(path))
            output_tokens_large = os.path.join(args.output, f"length-{args.tokens_split}-inf", os.path.basename(path))
            os.makedirs(os.path.dirname(output_tokens_small), exist_ok=True)
            os.makedirs(os.path.dirname(output_tokens_large), exist_ok=True)

            if os.path.exists(output_tokens_small + ".bin") or os.path.exists(output_tokens_large + ".bin"):
                continue

            builder_small = indexed_dataset.make_builder(
                output_tokens_small + ".bin", impl="mmap", vocab_size=vocab_size
            )
            builder_large = indexed_dataset.make_builder(
                output_tokens_large + ".bin", impl="mmap", vocab_size=vocab_size
            )

        try:
            dataset = indexed_dataset.MMapIndexedDataset(path)
            for i, doc in enumerate(dataset):
                if args.collect_stats:
                    _lengths.append(len(doc))
                    if max_len_for_stats and i >= max_len_for_stats:
                        break
                else:
                    if len(doc) <= args.tokens_split:
                        builder_small.add_doc(doc, [len(doc)])
                        stats = stats_small
                    else:
                        builder_large.add_doc(doc, [len(doc)])
                        stats = stats_large
                    ntokens = len(doc)
                    stats["total_tokens"] += ntokens
                    stats["total_sequences"] += 1
                    stats["min_tokens"] = min(stats["min_tokens"], ntokens)
                    stats["max_tokens"] = max(stats["max_tokens"], ntokens)

            # Finalize
            print()
            if not args.collect_stats:
                if stats_small["total_sequences"]:
                    builder_small.finalize(output_tokens_small + ".idx")
                    with open(output_tokens_small + ".json", "w") as f:
                        json.dump(stats_small, f, indent=2)
                else:
                    print(f"Warning: {output_tokens_small} is empty")
                    del builder_small
                    for ext in [".bin", ".idx"]:
                        if os.path.exists(output_tokens_small + ext):
                            os.remove(output_tokens_small + ext)

                if stats_large["total_sequences"]:
                    builder_large.finalize(output_tokens_large + ".idx")
                    with open(output_tokens_large + ".json", "w") as f:
                        json.dump(stats_large, f, indent=2)
                else:
                    print(f"Warning: {output_tokens_large} is empty")
                    del builder_large
                    for ext in [".bin", ".idx"]:
                        if os.path.exists(output_tokens_large + ext):
                            os.remove(output_tokens_large + ext)

        except (Exception, KeyboardInterrupt) as err:
            for f in [output_tokens_small, output_tokens_large]:
                for ext in [".bin", ".idx"]:
                    if os.path.exists(f + ext):
                        os.remove(f + ext)
            raise err

        if args.collect_stats:
            os.makedirs(os.path.dirname(output_length_stats), exist_ok=True)
            with open(output_length_stats, "w") as f:
                json.dump(_lengths, f, indent=2)
