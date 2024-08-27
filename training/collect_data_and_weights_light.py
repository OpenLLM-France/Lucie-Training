import csv
import json
import os
import re

import pandas as pd
import yaml

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
asset_folder = os.path.join(parent_dir, "assets")

_stats_datasets = os.path.join(asset_folder, "stats_datasets.csv")
assert os.path.exists(_stats_datasets), f"File {_stats_datasets} does not exist"


def read_stats_datasets(stats_datasets_filename=_stats_datasets):
    with open(stats_datasets_filename) as f:
        reader = csv.DictReader(f)
        stats_datasets = {}
        for d in reader:
            d = format_dictionary(d)
            if not d["name"].strip("-"):
                continue
            key = canonical_name(d["name"], d["subset"])
            stats_datasets[key] = d

    return stats_datasets


def format_programming_language(name):
    name = name.split("--")[-1]
    name = name.replace("_text_document", "")
    return name.lower()


def format_dictionary(d):
    d = {k.strip(): format_value(v) for k, v in d.items() if v}
    return d


def format_value(v):
    v = v.strip()
    for t in int, float:
        try:
            return t(v)
        except ValueError:
            pass
    return v


def canonical_name(name, subset=""):
    key = name.replace(".", "--") + "--" + subset
    key = key.rstrip("-")
    return key


def prefix_to_canonical_name(name, possible_names):  # noqa # C901 `...` is too complex
    name = os.path.basename(name)
    # name = os.path.splitext(name)[0]
    if name.endswith("_text_document"):
        name = name[: -len("_text_document")]
    if name not in possible_names:
        name2 = re.sub(r"\d+$", "", name)
        name2 = name2.rstrip("_").rstrip("-.")
        if name2 in possible_names:
            name = name2
    if name not in possible_names:
        if "--" in name:
            name2 = "--".join(name.split("--")[:-1])
            if name2 in possible_names:
                name = name2
            else:
                name2 = name.split("--")[0]
                if name2 in possible_names:
                    name = name2
        if name not in possible_names:
            name2 = re.sub(r"\.\d+$", "", name)
            if name2 in possible_names:
                name = name2
        if name not in possible_names:
            # Find optimal match based on edit distance
            best_match = None
            best_score = 1e32
            for possible_name in possible_names:
                try:
                    import editdistance

                    score = editdistance.eval(possible_name, name)
                except ImportError:
                    score = len([c for c in zip(possible_name, name) if c[0] != c[1]])
                if best_match is None or score < best_score:
                    best_match = possible_name
                    best_score = score
            print(f"WARNING: Dataset {name} not found: {best_match=}")
    return name


if __name__ == "__main__":
    import argparse

    default_path = "/data-storage/storage0/lucie_tokens_65k_grouped"
    for path in [
        "/data-storage/storage0/lucie_tokens_65k_grouped",
        "/lustre/fsn1/projects/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_65k_grouped",
    ]:
        if os.path.exists(path):
            default_path = path
            break

    parser = argparse.ArgumentParser(
        description="Prints a string with all tokenized data files (prefixes) and their respective weights.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "folder",
        type=str,
        help="Path to tokenized data",
        default=default_path,
        nargs="?",
    )
    parser.add_argument(
        "--count",
        type=str,
        default="total_tokens",
        help="What to count",
    )
    parser.add_argument(
        "--fr_weight",
        type=float,
        default=1.0,
        help="How much French data in total",
    )
    parser.add_argument(
        "--en_weight",
        type=float,
        default=1.0,
        help="How much English data in total",
    )
    parser.add_argument(
        "--code_weight",
        type=float,
        default=1.0,
        help="How much Code data in total",
    )
    parser.add_argument(
        "--other_weight",
        type=float,
        default=1.0,
        help="How much Code data in total",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="To print debug output",
    )
    args = parser.parse_args()

    stats_datasets = read_stats_datasets()
    add_language_weights = {
        "en": args.en_weight,
        "fr": args.fr_weight,
        "code": args.code_weight,
    }

    with open(os.path.join(asset_folder, "dataset_weights.yaml")) as stream:
        domain_upsampling = yaml.safe_load(stream)

    not_tokenized_datasets = list(stats_datasets.keys())

    prefixes = []
    for filename in sorted(os.listdir(args.folder)):
        if not filename.endswith(".idx"):
            continue
        prefixes.append(os.path.splitext(filename)[0])

    data = {}

    for prefix in prefixes:
        prefix = os.path.join(args.folder, prefix)

        name = prefix_to_canonical_name(prefix, stats_datasets)
        if name not in stats_datasets:
            raise RuntimeError(f"Dataset {name} cannot be matched ({prefix=}, {sorted(stats_datasets.keys())=})")
        if name in not_tokenized_datasets:
            not_tokenized_datasets.remove(name)

        def load_data_from_prefix(prefix):
            json_filename = os.path.join(args.folder, prefix + ".json")
            if not os.path.exists(json_filename):
                raise RuntimeError(f"File {json_filename} does not exist")
            with open(json_filename) as f:
                d = json.load(f)
            return d

        d = load_data_from_prefix(prefix)
        d.update(stats_datasets[name])

        num_epochs = domain_upsampling[d["language"] + "--" + d["category"]] * add_language_weights.get(
            d["language"], args.other_weight
        )
        count = d[args.count]
        reweighted_count = num_epochs * count

        d["num_epochs"] = num_epochs
        d["count"] = count
        d["reweighted_count"] = reweighted_count

        data[prefix] = d

    if not_tokenized_datasets and args.debug:
        print(f"WARNING! Those datasets are missing (not tokenized): {', '.join(not_tokenized_datasets)}")

    # Convert to pandas
    df = pd.DataFrame.from_dict(data, orient="index").reset_index(names="prefix")
    total_count = df["count"].sum()
    total_reweighted_count = df["reweighted_count"].sum()

    # Weight per dataset
    df["ratio"] = df["count"] / total_count
    df["new_ratio"] = df["reweighted_count"] / total_reweighted_count
    df = df.sort_values("reweighted_count", ascending=False)

    if args.debug:
        print("# Weights per sub-corpus\n```")
        for _, row in df.iterrows():
            name = row["prefix"].split("/")[-1][: -len("_text_document")]
            ratio = row["ratio"]
            new_ratio = row["new_ratio"]
            num_epochs = row["num_epochs"]
            reweighted_count = row["reweighted_count"]
            print(
                f"{name:40s}: \
before={ratio*100:6.3f}% after={new_ratio*100:6.3f}% reweighted_count={reweighted_count*1e-9:.3f} B tokens \
-> num_epochs={num_epochs:3.1f}"
            )
        print("```\n")

        print("# Weights per language\n```")
        df["language"] = df["language"].apply(lambda x: x if x in ["en", "fr", "de", "es", "it", "code"] else "aligned")
        df_lan = df.groupby("language")[["count", "reweighted_count"]].sum().reset_index()
        df_lan["ratio"] = df_lan["count"] / total_count
        df_lan["new_ratio"] = df_lan["reweighted_count"] / total_reweighted_count
        df_lan = df_lan.sort_values("reweighted_count", ascending=False)

        for _, row in df_lan.iterrows():
            language = row["language"]
            reweighted_count = row["reweighted_count"]
            ratio = row["ratio"]
            new_ratio = row["new_ratio"]

            print(
                f"{language:40s}: \
before={ratio*100:6.3f}% after={new_ratio*100:6.3f}% reweighted_count={reweighted_count*1e-9:.3f} B tokens"
            )
        print("```\n")

        print("# Total Tokens\n```")
        print(f"before={total_count*1e-9:.3f} B tokens, after={total_reweighted_count*1e-9:.3f} B tokens")
        print(f"Number of samples={total_reweighted_count/4096:.2f} with sequence length=4096")
        print("```")

    else:
        for _, row in df.iterrows():
            prefix = row["prefix"]
            new_ratio = row["new_ratio"]
            # Print the weight (expected output)
            sweight = f"{new_ratio:11.9f}"
            print(f"{sweight} {prefix} ", end="")

            # Check that nothing was rounded to weight=0
            if not re.search(r"[^\.0]", sweight):
                print()
                raise RuntimeError(f"Weight is zero for {prefix}")
