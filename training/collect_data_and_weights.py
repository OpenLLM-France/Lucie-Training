import csv
import json
import os
import re

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
asset_folder = os.path.join(parent_dir, "assets")

stats_datasets = os.path.join(asset_folder, "stats_datasets.csv")
assert os.path.exists(stats_datasets), f"File {stats_datasets} does not exist"


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
    name = re.sub(r"\d+$", "", name)
    name = name.rstrip("_").rstrip("-")
    if name not in possible_names:
        if "--" in name:
            name2 = "--".join(name.split("--")[:-1])
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


def read_stats_datasets(stats_datasets_filename=stats_datasets):
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prints a string with all tokenized data files (prefixes) and their respective weights.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "folder",
        type=str,
        help="Path to tokenized data",
        default="/data-storage/storage0/lucie_tokens_2.9",
        nargs="?",
    )
    parser.add_argument(
        "--count",
        type=str,
        default="total_tokens",
        help="What to count",
    )
    parser.add_argument(
        "--wikipedia_weight",
        type=float,
        default=5,
        help="How much to weight Wikipedia (like duplicating). 1 meaning no change.",
    )
    parser.add_argument(
        "--fr_proportion",
        type=float,
        default=0.3,
        help="How much French data in total",
    )
    parser.add_argument(
        "--en_proportion",
        type=float,
        default=0.3,
        help="How much English data in total",
    )
    parser.add_argument(
        "--code_proportion",
        type=float,
        default=0.3,
        help="How much Code data in total",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="To print debug output",
    )
    args = parser.parse_args()

    language_target_proportions = {
        "fr": args.fr_proportion,
        "en": args.en_proportion,
        "code": args.code_proportion,
        # The rest (10% will be splitted among it/es/de)
    }

    additional_weights = {
        "Wikipedia": args.wikipedia_weight,
    }

    stats_datasets = read_stats_datasets()

    # import json
    # print(json.dumps(stats_datasets, indent=4))

    not_tokenized_datasets = list(stats_datasets.keys())

    prefixes = []
    for filename in sorted(os.listdir(args.folder)):
        if not filename.endswith(".idx"):
            continue
        prefixes.append(os.path.splitext(filename)[0])

    data = {}
    num_tokens_per_language = {}
    num_tokens_per_language_weighted = {}

    for prefix in prefixes:
        prefix = os.path.join(args.folder, prefix)

        name = prefix_to_canonical_name(prefix, stats_datasets)
        if name not in stats_datasets:
            raise RuntimeError(f"Dataset {name} cannot be matched ({prefix=})")
            continue
        if name in not_tokenized_datasets:
            not_tokenized_datasets.remove(name)

        json_filename = prefix + ".json"
        if not os.path.exists(json_filename):
            raise ValueError(f"File {json_filename} does not exist")
        with open(json_filename) as f:
            d = json.load(f)

        d.update(stats_datasets[name])
        data[prefix] = d

        additional_weight = 1
        for content in additional_weights:
            if re.search(content, prefix):
                additional_weight *= additional_weights[content]

        languages = d["language"].split("-")
        count = d[args.count]
        count_weighted = additional_weight * count
        for language in languages:
            num_tokens_per_language_weighted[language] = num_tokens_per_language_weighted.get(language, 0) + (
                count_weighted // len(languages)
            )
            num_tokens_per_language[language] = num_tokens_per_language.get(language, 0) + (count // len(languages))

    if not_tokenized_datasets and args.debug:
        print(f"WARNING! Those datasets are missing (not tokenized): {', '.join(not_tokenized_datasets)}")

    total_count = sum(num_tokens_per_language.values())
    total_count_weighted = sum(num_tokens_per_language_weighted.values())
    total_count_weighted_rest = total_count_weighted - sum(
        [num_tokens_per_language_weighted.get(lan, 0) for lan in language_target_proportions]
    )

    language_target_proportion_rest = 1 - sum(language_target_proportions.values())
    assert (
        language_target_proportion_rest >= 0 and language_target_proportion_rest < 1
    ), f"{language_target_proportion_rest=}"

    language_weights = {}
    for language, count_weighted in num_tokens_per_language_weighted.items():
        if language in language_target_proportions:
            language_target_proportion = language_target_proportions[language]
        else:
            language_target_proportion = language_target_proportion_rest * count_weighted / total_count_weighted_rest
            language_target_proportions[language] = language_target_proportion
        weight = language_target_proportion / (count_weighted / total_count_weighted)
        language_weights[language] = weight

    if args.debug:
        total_count_weighted_weighted = sum(
            num_tokens_per_language_weighted[language] * language_weights[language]
            for language in num_tokens_per_language_weighted
        )
        for language, count in num_tokens_per_language_weighted.items():
            language_target_proportion = language_target_proportions[language]
            weight = language_weights[language]
            print(
                f"Language weight: {language=:4s} {language_target_proportion=:4.3f} {weight=:7.6f}\
before={count * 100/ total_count_weighted:6.3f}% after={count * weight * 100/ total_count_weighted_weighted:6.3f}%"
            )

    for normalize_weight in [False, True]:
        if not normalize_weight:
            all_weights = []
            total_weights = 0
            norm_weight = 0

        for prefix, d in data.items():
            languages = d["language"].split("-")
            count = d[args.count]
            ratio = count / total_count

            language_weight = max(language_weights[language] for language in languages)

            additional_weight = 1
            for content in additional_weights:
                if re.search(content, prefix):
                    additional_weight *= additional_weights[content]

            weight = ratio * language_weight * additional_weight

            all_weights.append(weight)
            if normalize_weight:
                new_ratio = weight / total_weights

                weight = weight / norm_weight

                if args.debug:
                    name = os.path.basename(prefix).replace("_text_document", "")
                    print(
                        f"{name:50s}: {language_weight=:7.6f} {additional_weight=:7.6f} {weight=:7.6f}\
before={ratio * 100:5.3f}% after={new_ratio * 100:5.3f}%"
                    )

                else:
                    print(f"{weight:10.9f} {prefix} ", end="")
            else:
                # Median weight is 1
                # norm_weight = sorted(all_weights)[len(all_weights)//2]
                norm_weight = 1 / 1000
                total_weights += weight

    if not args.debug:
        print()
