import csv
import json
import os
import re
import warnings
import yaml

import pandas as pd

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
asset_folder = os.path.join(parent_dir, "assets")

_stats_datasets = os.path.join(asset_folder, "stats_datasets.csv")
assert os.path.exists(_stats_datasets), f"File {_stats_datasets} does not exist"

_stats_programming_languages = {
    stat_name: os.path.join(
        asset_folder, "programming-languages", "githut", f"gh-{stat_name}.json"
    )
    for stat_name in ["pull-request", "issue-event", "star-event", "push-event"]
}
_minimum_count = None


def get_programming_language_stat(
    language,
    stat_name="pull-request",
    min_year=2023,
    max_year=2024,
    no_minimum_count=False,
):
    global _stats_programming_languages, _minimum_count
    assert (
        stat_name in _stats_programming_languages
    ), f"Unknown statistic name {stat_name}"
    language = format_programming_language(language)

    data = _stats_programming_languages[stat_name]
    if isinstance(data, str):
        assert os.path.exists(data), f"File {data} does not exist"
        # First time loading
        with open(data) as f:
            data = _stats_programming_languages[stat_name] = json.load(f)
    if isinstance(data, list):
        # Conversion
        data = pd.DataFrame(data)
        data["name"] = data["name"].apply(format_programming_language)
        data["year"] = data["year"].apply(int)
        data["quarter"] = data["quarter"].apply(int)
        data["count"] = data["count"].apply(int)
        _stats_programming_languages[stat_name] = data

    data = data[(data["year"] >= min_year) & (data["year"] <= max_year)]
    val = data[(data["name"] == language)]
    if not len(val):
        if no_minimum_count:
            return 0
        if _minimum_count is None:
            _minimum_count = min(
                [
                    get_programming_language_stat(
                        lan,
                        stat_name=stat_name,
                        min_year=min_year,
                        max_year=max_year,
                        no_minimum_count=True,
                    )
                    for lan in data["name"].unique()
                ]
            )
        warnings.warn(
            f"Programming language {language} not found in statistics (using {_minimum_count=})",
            stacklevel=2,
        )
        return _minimum_count
    return val["count"].sum()


def compute_programming_languages_target_proportions(programming_languages):
    programming_languages_weights = {}
    for language in programming_languages:
        count = get_programming_language_stat(language)
        programming_languages_weights[language] = count
    total = sum(programming_languages_weights.values())
    programming_languages_weights = {
        k: v / total for k, v in programming_languages_weights.items()
    }
    return programming_languages_weights


def read_stats_datasets(stats_datasets_filename=_stats_datasets):
    """Reads the stats datasets file and returns a dictionary with the dataset names as keys.

    The dictionary contains the following
    - name: the dataset name
    - category: the dataset category
    - subset: the dataset subset
    - ocr: the OCR value
    - M docs: the number of documents in millions
    - B words: the number of words in billions
    - B chars: the number of characters in billions
    - #words/doc: the average number of words per document
    - #chars/word: the average number of characters per word
    - B tokens: the number of tokens in billions
    - #tokens/words: the average number of tokens per word
    - #chars/tokens: the average number of characters per token

    """
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
    name = re.sub(r"\d+$", "", name)
    name = name.rstrip("_").rstrip("-.")
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

    parser = argparse.ArgumentParser(
        description="Prints a string with all tokenized data files (prefixes) and their respective weights.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "folder",
        type=str,
        help="Path to tokenized data",
        default="/data-storage/storage0/lucie_tokens_65k_grouped",
        nargs="?",
    )
    parser.add_argument(
        "--count",
        type=str,
        default="total_tokens",
        help="What to count",
    )
    parser.add_argument(
        "--domain_proportions",
        type=str,
        default="domain_proportions.yml",
        help="Path to the domain proportions configuration",
    )
    parser.add_argument(
        "--save_weights_path",
        type=str,
        default=None,
        help="Path to save the weights configuration",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="To print debug output",
    )
    args = parser.parse_args()

    with open(args.domain_proportions, "r") as stream:
        domain_target_proportions = yaml.safe_load(stream)

    additional_weights = {}

    if args.save_weights_path:
        proportion_dict = {
            "domain_target_proportions": domain_target_proportions,
            "additional_weights": additional_weights,
        }
        with open(f"{args.save_weights_path}/proportion_args.json", "w") as f:
            f.write(json.dumps(proportion_dict, indent=4))

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
    num_tokens_per_domain = {}
    num_tokens_per_domain_weighted = {}
    num_tokens_per_programming_language = {}

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
            raise RuntimeError(f"File {json_filename} does not exist")
        with open(json_filename) as f:
            d = json.load(f)

        d.update(stats_datasets[name])
        data[prefix] = d

        additional_weight = 1
        for content in additional_weights:
            if re.search(content, prefix):
                additional_weight *= additional_weights[content]

        domains = d["category"].split("-")
        count = d[args.count]
        count_weighted = additional_weight * count
        for domain in domains:
            num_tokens_per_domain_weighted[domain] = num_tokens_per_domain_weighted.get(
                domain, 0
            ) + (count_weighted // len(domains))
            num_tokens_per_domain[domain] = num_tokens_per_domain.get(domain, 0) + (
                count // len(domains)
            )

        if domain == "code":
            prog_lang = format_programming_language(name)
            num_tokens_per_programming_language[prog_lang] = (
                num_tokens_per_programming_language.get(prog_lang, 0) + count
            )
    print(json.dumps(data, indent=4))

    if not_tokenized_datasets and args.debug:
        print(
            f"WARNING! Those datasets are missing (not tokenized): {', '.join(not_tokenized_datasets)}"
        )

    # Sort data by count
    data = {
        k: v
        for k, v in sorted(
            data.items(), key=lambda item: item[1][args.count], reverse=True
        )
    }  # noqa

    total_count = sum(num_tokens_per_domain.values())
    total_count_weighted = sum(num_tokens_per_domain_weighted.values())
    total_count_weighted_rest = total_count_weighted - sum(
        [
            num_tokens_per_domain_weighted.get(dom, 0)
            for dom in domain_target_proportions
        ]
    )

    domain_target_proportion_rest = 1 - sum(domain_target_proportions.values())
    assert (
        domain_target_proportion_rest >= 0 and domain_target_proportion_rest < 1
    ), f"{domain_target_proportion_rest=}"

    # Set the weights for domain (newspaper, book, code, ...)
    domain_weights = {}
    for domain, count_weighted in num_tokens_per_domain_weighted.items():
        if domain in domain_target_proportions:
            target_proportion = domain_target_proportions[domain]
        else:
            target_proportion = (
                domain_target_proportion_rest
                * count_weighted
                / total_count_weighted_rest
            )
            domain_target_proportions[domain] = target_proportion
        weight = target_proportion / (count_weighted / total_count_weighted)
        domain_weights[domain] = weight

    # Set the weights for programming languages
    programming_language_target_proportions = (
        compute_programming_languages_target_proportions(
            num_tokens_per_programming_language.keys()
        )
    )
    programming_language_weights = {}
    for language, count_weighted in num_tokens_per_programming_language.items():
        assert (
            language in programming_language_target_proportions
        ), f"{language=} not found"
        target_proportion = (
            programming_language_target_proportions[language]
            * domain_target_proportions["code"]
        )
        weight = target_proportion / (count_weighted / total_count_weighted)
        programming_language_weights[language] = weight

    if args.debug:
        for what, lf, num_tokens, target_proportions, weights in [
            (
                "domain",
                "4s",
                num_tokens_per_domain_weighted,
                domain_target_proportions,
                domain_weights,
            ),
            (
                "programming language",
                "12s",
                num_tokens_per_programming_language,
                programming_language_target_proportions,
                programming_language_weights,
            ),
        ]:
            print(f"# Weights per {what}\n```")
            num_tokens = {  # noqa
                k: v
                for k, v in sorted(
                    num_tokens.items(), key=lambda item: item[1], reverse=True
                )
            }
            total_tokens_weighted = sum(
                num_tokens[domain] * weights[domain] for domain in num_tokens
            )
            total_tokens = sum(num_tokens.values())
            for domain, count in num_tokens.items():
                target_proportion = target_proportions[domain]
                weight = weights[domain]
                domain = domain.format("")
                print(
                    f"{domain=:{lf}} {target_proportion=:4.3f} {weight=:9.6f} \
before={count * 100/ total_tokens:6.3f}% after={count * weight * 100/ total_tokens_weighted:6.3f}%"
                )
            print("```\n")

        print("# Weights per sub-domain\n```")

    for second_pass in [False, True]:
        if not second_pass:
            all_weights = {}
        else:
            data = {
                k: v
                for k, v in sorted(
                    data.items(), key=lambda x: all_weights[x[0]], reverse=True
                )
            }  # noqa
            total_weights = sum(all_weights.values())
            # Normalization factor for weights
            norm_weight = total_weights / 100

        if args.save_weights_path:
            final_weight_dict = {}

        for prefix, d in data.items():
            domains = d["category"].split("-")
            count = d[args.count]
            ratio = count / total_count

            domain_weight = max(domain_weights[domain] for domain in domains)
            if d["category"] == "code":
                prog_language = format_programming_language(prefix)
                domain_weight = programming_language_weights[prog_language]

            additional_weight = 1
            for content in additional_weights:
                if re.search(content, prefix):
                    additional_weight *= additional_weights[content]

            weight = all_weights[prefix] = ratio * domain_weight * additional_weight

            if second_pass:
                new_ratio = weight / total_weights
                weight = weight / norm_weight

                if args.debug:
                    name = os.path.basename(prefix).replace("_text_document", "")
                    print(
                        f"{name:40s}: {weight=:12.9f} \
before={ratio * 100:6.3f}% after={new_ratio * 100:6.3f}% ({domain_weight=:8.6f} {additional_weight=:3.1f})"
                    )

                else:
                    # Print the weight (expected output)
                    sweight = f"{weight:11.9f}"
                    print(f"{sweight} {prefix} ", end="")
                    if args.save_weights_path:
                        final_weight_dict[prefix] = weight
                    # Check that nothing was rounded to weight=0
                    if not re.search(r"[^\.0]", sweight):
                        print()
                        raise RuntimeError(f"Weight is zero for {prefix}")

    if args.save_weights_path:
        with open(f"{args.save_weights_path}/final_weights.json", "w") as f:
            f.write(json.dumps(final_weight_dict, indent=4))
    if args.debug:
        print("```")
    else:
        print()

# DATASET="$(python ~/Lucie-Training/training/collect_data_and_weights_alt.py /local_data/lucie_tokens_65k_grouped)"
