import json
import os
import sys

import tqdm

rootdir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
megatron_deepspeed_folder = os.path.join(rootdir, "Megatron-DeepSpeed")
sys.path = [megatron_deepspeed_folder] + sys.path  # Better to prepend for "tools" module

from megatron.data import indexed_dataset  # noqa # E402 Module level import not at top of file


def get_name(dataset):
    name = os.path.basename(dataset)
    f = name.split("--")
    if f[1] in ["en", "fr", "de", "es", "it"]:
        name = "--".join(f[:2])
    else:
        name = f[0]
    return name


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Count tokens in indexed datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "folder",
        type=str,
        help="Folder with indexed datasets",
        default="/data-storage/storage0/lucie_tokens_2.9",
        nargs="?",
    )
    args = parser.parse_args()

    folder = args.folder
    save_in_json = True

    total_num_tokens = {}

    paths = []

    for file in os.listdir(folder):
        if not file.endswith(".idx"):
            continue
        path = os.path.join(folder, os.path.splitext(file)[0])
        paths.append(path)

    for path in tqdm.tqdm(sorted(paths)):
        name = get_name(path)
        json_file = None
        if save_in_json:
            json_file = path + ".json"
        if json_file and os.path.exists(json_file):
            data = json.load(open(json_file))
            total_tokens = data["total_tokens"]
            total_sequences = data["total_sequences"]
            min_tokens = data["min_tokens"]
            max_tokens = data["max_tokens"]
        else:
            dataset = indexed_dataset.MMapIndexedDataset(path)
            total_tokens = 0
            total_sequences = 0
            min_tokens = 1e32
            max_tokens = 0
            for data in dataset:
                n = len(data)
                total_tokens += n
                min_tokens = min(min_tokens, n)
                max_tokens = max(max_tokens, n)
                total_sequences += 1
            if json_file and save_in_json:
                json.dump(
                    {
                        "total_tokens": total_tokens,
                        "total_sequences": total_sequences,
                        "min_tokens": min_tokens,
                        "max_tokens": max_tokens,
                    },
                    open(json_file, "w"),
                )

        total_num_tokens["TOTAL"] = total_num_tokens.get("TOTAL", 0) + total_tokens
        total_num_tokens[name] = total_num_tokens.get(name, 0) + total_tokens

    for k in total_num_tokens:
        total_num_tokens[k] /= 1e9

    s = json.dumps(total_num_tokens, indent=4)
    print(s)
    print(s, file=open("total_num_tokens.json", "w"))
