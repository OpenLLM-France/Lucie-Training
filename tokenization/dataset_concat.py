import json
import os
import re
import shutil
import sys

import tqdm

rootdir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
megatron_deepspeed_folder = os.path.join(rootdir, "Megatron-DeepSpeed")
sys.path = [megatron_deepspeed_folder] + sys.path  # Better to prepend for "tools" module

from megatron.data import indexed_dataset  # noqa # E402 Module level import not at top of file

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Concatenate MMap Indexed datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("inputs", type=str, nargs="+", help="Input indexed dataset filenames (without extension)")
    parser.add_argument("output", type=str, help="Output filenames prefix (no extension)")
    parser.add_argument(
        "--vocab_size", type=int, default=65024, help="Vocabulary size (is it larger or not than 65500?)"
    )
    parser.add_argument(
        "--clean_inputs", action="store_true", default=False, help="Remove input files after concatenation"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", default=False, help="Verbose mode, where it tells what it's doing"
    )
    parser.add_argument("--only", default=False, help="Only process when output match this regex")
    parser.add_argument(
        "--dry-run", "-n", action="store_true", default=False, help="Show what would be done without doing it"
    )
    args = parser.parse_args()
    vocab_size = args.vocab_size

    if len(args.inputs) == 1 and os.path.isdir(args.inputs[0]):
        output_to_inputs = {}
        input_folder = args.inputs[0]
        output_folder = args.output
        for file in sorted(os.listdir(input_folder)):
            if not file.endswith(".idx"):
                continue
            input_path = os.path.join(input_folder, file)
            if "AmericanStories" in input_path:
                output_path = re.sub(r"[\-\d]*\d([a-zA-Z_]+).idx$", r"\1", input_path)
            else:
                output_path = re.sub(r"\-*\d*\d([a-zA-Z_]+).idx$", r"\1", input_path)
            input_path = os.path.splitext(input_path)[0]
            if output_path.endswith(".idx"):
                output_path = os.path.splitext(output_path)[0]
            output_path = os.path.join(output_folder, os.path.basename(output_path))
            if args.only and args.only not in output_path:
                continue
            output_to_inputs.setdefault(output_path, []).append(input_path)

        outputs = sorted(output_to_inputs.keys())
        inputs_lists = [output_to_inputs[output] for output in outputs]
    else:
        inputs_lists = [args.inputs]
        outputs = [args.output]

    for inputs, output in zip(inputs_lists, tqdm.tqdm(outputs)):
        assert len(inputs)
        assert output
        assert output not in inputs

        if args.dry_run or args.verbose:
            action = "Concatenate"
            plural = "s"
            if len(inputs) == 1:
                action = "Move" if args.clean_inputs else "Copy"
                plural = ""
            print(f"{action} {len(inputs)} file{plural} into {output} ({sorted(inputs)[0]}... -> {output})")
            if args.dry_run:
                continue

        # Make sure output folder exists
        dirname = os.path.dirname(output)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)

        if len(inputs) == 1:
            # Simple copy
            input = inputs[0]
            for ext in [".idx", ".bin", ".json"]:
                if os.path.exists(input + ext):
                    if args.clean_inputs:
                        shutil.move(input + ext, output + ext)
                    else:
                        shutil.copy2(input + ext, output + ext)
            continue

        output_bin_file = output + ".bin"
        output_idx_file = output + ".idx"

        # Aggregate data and write output bin
        builder = indexed_dataset.make_builder(output_bin_file, impl="mmap", vocab_size=vocab_size)

        try:
            for path in inputs:
                dataset = indexed_dataset.MMapIndexedDataset(path)
                for doc in dataset:
                    builder.add_doc(doc, [len(doc)])
        except (Exception, KeyboardInterrupt) as err:
            if os.path.exists(output_bin_file):
                os.remove(output_bin_file)
            raise err

        builder.finalize(output_idx_file)

        # Aggregate statistics in json files
        if all(os.path.exists(input + ".json") for input in inputs):
            # Merge json files
            data = None
            for input in inputs:
                with open(input + ".json") as f:
                    d = json.load(f)
                if data is None:
                    data = d
                else:
                    for k, v in d.items():
                        assert k in data
                        if k.startswith("min"):
                            data[k] = min(data[k], v)
                        elif k.startswith("max"):
                            data[k] = max(data[k], v)
                        else:
                            assert k.startswith("total")
                            data[k] += v
            with open(output + ".json", "w") as f:
                json.dump(data, f, indent=2)

        if args.clean_inputs:
            for input in inputs:
                for ext in [".idx", ".json", ".bin"]:
                    if os.path.exists(input + ext):
                        os.remove(input + ext)
