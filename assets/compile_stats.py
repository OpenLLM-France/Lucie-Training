import csv
import json
import os


# Ignore datasets
def ignore_datasets(name):
    return "subscene" in name.lower()  # or redpajama" in name.lower()


text_types = {
    "ocr": [
        "AmericanStories",
        "Eurovoc",
        "GallicaPress",
        "GallicaMonographies",
        "HAL",
        # 'OtherFr',
        "Persee",
        "Theses",
    ],
    "mixed": ["PeS2o"],
}

datasets_categories = {
    "technical": [
        "HAL",
        "NIH_ExPorter",
        "OpenEdition",
        "Persee",
        "PeS2o",
        "PhilPapers",
        "Theses",
        "USPTO_Backgrounds",
    ],
    "legi_written": [
        "amendements_parlement",
        "Eurovoc.de",
        "Eurovoc.en",
        "Eurovoc.es",
        "Eurovoc.it",
        "FreeLaw",
        "LEGI",
        "OpenData",
        "questions_ecrites_parlement",
    ],
    "legi_spoken": [
        "DiscoursPublics",
        "interventions_parlement",
    ],
    "legi_dialogue": [
        "AssembleeNationale",
        "Europarl.en",
        "Europarl.es",
        "Europarl.de",
        "Europarl.fr",
        "FREDSum",
        "Senat",
    ],
    "dialogue": [
        "Claire.en",
        "Claire.fr",
        "Stac",
        "ValidatedYouTube.fr",
    ],
    "book": [
        "GallicaMonographies",
        "Gutenberg.en",
        "Gutenberg.de",
        "Gutenberg.it",
        "Gutenberg.es",
        "Gutenberg.fr",
    ],
    "newspaper": [
        "AmericanStories",
        "GallicaPress",
    ],
    "forum": [
        "StackExchange",
        "Ubuntu_IRC",
    ],
    "wiki": [
        "Wikiother.fr",
        "Wikipedia.en",
        "Wikipedia.es",
        "Wikipedia.de",
        "Wikipedia.it",
        "Wikipedia.fr",
    ],
    "programming": [
        "TheStack",
    ],
    "math": [
        "DM_Mathematics",
        "MathPile",
    ],
    "aligned": [
        "CroissantAligned",
        "EuroparlAligned.fr-en",
        "EuroparlAligned.es-en",
        "EuroparlAligned.it-en",
        "EuroparlAligned.de-fr",
    ],
    "web": [
        "culturax.fr",
        "culturax.en",
        "culturax.de",
        "culturax.es",
        "culturax.it",
        "RedPajama.fr",
        "RedPajama.en",
        "RedPajama.de",
        "RedPajama.es",
        "RedPajama.it",
        "FineWebEdu--cc-main-2021",
        "FineWebEdu--cc-main-2022",
        "FineWebEdu--cc-main-2023",
        "FineWebEdu--cc-main-2024",
        "FineWebEdu--cc-main",
        "FineWebEdu",
    ],
}


def _norm_string(s):
    return s.lower().replace("_", " ")


text_types = {k: [_norm_string(x) for x in v] for k, v in text_types.items()}
datasets_categories = {k: [_norm_string(x) for x in v] for k, v in datasets_categories.items()}


def is_ocr_dataset(name, subset):
    if name in ["---", "", None]:
        return ""
    name = _norm_string(name)
    res = "false"
    if any(d in name for d in text_types["ocr"]):
        res = "true"
    if any(d in name for d in text_types["mixed"]):
        res = "mixed"
    return res


def get_dataset_category(name, subset):
    if name in ["---", "", None]:
        return ""
    name = _norm_string(name)
    if name in ["pile", "otherfr"] and subset:
        name = _norm_string(subset)
    for cat, datasets in datasets_categories.items():
        if name in datasets:
            return cat
    print(f"WARNING: category not found for {name}")
    return None


def to_name_subset(name):
    orig_name = name
    name = name.replace("stats_", "").replace(".json", "").replace("_text_document", "")
    subset = ""
    if "--" in name:
        name = name.replace("--", ":")
        name, subset = name.split(":", 1)
    if subset:
        subset = subset.replace("_opendata", "").lstrip(":")
    if "train" in subset and "cultura" in name.lower():
        subset = subset.replace("train", "")
    if subset:
        subset = subset.strip(":_")
    if "fineweb" in name.lower():
        assert subset.startswith("cc-main-"), f"Invalid FineWeb subset {subset=} {orig_name=}"
        subset = subset.replace("cc-main-", "")
        f = subset.split(":")[-1].split("-")
        f[0] = "cc-main-" + f[0]
        name += "--" + f[0]
        if len(f) == 1:
            subset = ""
    if "redpajama" in name.lower():
        f = subset.split("-")
        if len(f) == 1:
            subset = ""
    return name, subset


def to_language_name_subset(name, subset=None):  # noqa # C901 `...` is too complex
    if subset is None:
        name, subset = to_name_subset(name)
    for lan in "fr", "en", "de", "es", "it":
        subset2 = subset.rstrip(":.0123456789")
        if subset.startswith(lan) and (len(subset) == len(lan) or subset[len(lan)] in ".:-"):
            if "-" in subset and len(subset2) == 5:
                subset = subset2
                lan = subset
            subset = subset[len(lan) :].strip(":.")
            subset = subset.strip(":_")
            if "gutenberg" in name.lower():
                subset = ""
            return lan, name + "." + lan, subset
    if "En" in name:
        language = "en"
    elif any(x in name.lower() for x in ["americanstories", "pes2o"]):
        language = "en"
    elif "De" in name:
        language = "de"
    elif "Es" in name:
        language = "es"
    elif "It" in name:
        language = "it"
    elif "TheStack" in name:
        language = "code"
    elif "Pile" in name or "Stac" in name or "FineWebEdu" in name:  # Pile and MathPile
        language = "en"
    else:
        language = "fr"
    # Multi-lingual corpora
    if name in [
        "Claire",
        "Wikipedia",
        "Europarl",
        "Gutenberg",
        "Wikiother",
        "Eurovoc",
        "EuroparlAligned",
        "CulturaX",
        "RedPajama",
    ]:
        language = None
    if name == "CroissantAligned":
        language = "fr-en"
    if "Wikipedia" in name and "Europarl" in name:
        language = None

    if subset:
        subset = subset.strip(":").replace(":", "-")
        subset = subset.split(":")[0]
        subset = subset.replace("train-pile_", "")
        subset = subset.strip(":_")

    return language, name, subset


KEYS = {
    "num pages": "#docs",
    "num words": "#words",
    "num chars": "#chars",
}

SORT_BY = "#words"


def compute_extra_stats(data, tokencount_folder):
    if "B words" in data:
        return data
    try:
        assert "#words" in data, f"Missing #words in {data.keys()}"

        data["ocr"] = is_ocr_dataset(data.get("name", ""), data.get("subset"))
        data["category"] = get_dataset_category(data.get("name", ""), data.get("subset"))

        data["M docs"] = data["#docs"] / 1_000_000
        data["B words"] = data["#words"] / 1_000_000_000
        data["B chars"] = data["#chars"] / 1_000_000_000
        data["#words/doc"] = data["#words"] / (data["#docs"] or 1)
        # data["#chars/page"] = data["#chars"] / data["#docs"]
        data["#chars/word"] = data["#chars"] / (data["#words"] or 1)

        if tokencount_folder:
            global total_tokens
            key = (data.get("language"), data.get("name"), data.get("subset"))
            key = to_dict_key(key)
            data["B tokens"] = None
            data["#tokens/words"] = None
            data["#chars/tokens"] = None
            if key in total_tokens:
                data["B tokens"] = total_tokens[key] / 1_000_000_000
                if abs(data["M docs"] - total_sequences_check[key] / 1_000_000) > 0.01:
                    print(
                        f"WARNING: mismatch for {key}: \
(stats) {data['M docs']} != (token) {total_sequences_check[key] / 1_000_000}"
                    )
                else:
                    data["#tokens/words"] = data["B tokens"] / data["B words"]
                    data["#chars/tokens"] = data["#chars"] / total_tokens[key]
            elif key == to_dict_key(("---", "---", "---")):
                data["B tokens"] = ground_total_tokens / 1_000_000_000
            elif not data.get("subset"):
                print(f"WARNING: missing {key} in tokens")

        data.pop("#docs")
        data.pop("#words")
        data.pop("#chars")

    except Exception as err:
        raise RuntimeError(f"Error processing {data}") from err
    return data


def get_stat_names(compute_token_stats=True):
    dummy = {k: 1 for k in KEYS.values()}
    dummy = compute_extra_stats(dummy, compute_token_stats)
    return list(dummy.keys())


def format_stats_display(data, main=True):
    for name, format in [
        ("language", "{:<9s}"),
        ("name", "{:<21s}"),
        ("subset", "{:<12s}" if main else "{:<28s}"),
        ("ocr", "{:<6s}"),
        ("category", "{:<13s}"),
        ("M docs", "{:8.3f}"),
        ("B words", "{:8.3f}"),
        ("B chars", "{:8.3f}"),
        ("B tokens", "{:9.3f}"),
        ("#words/doc", "{:11.0f}"),
        ("#chars/page", "{:11.0f}"),
        ("#chars/word", "{:11.1f}"),
        ("#tokens/words", "{:12.2f}"),
        ("#chars/tokens", "{:12.2f}"),
    ]:
        if name in data.keys():
            val = data[name]
            if isinstance(val, str) or val is None:
                if val is None:
                    val = " "
                if format.endswith("f}"):
                    length = int(format[2:-2].split(".")[0])
                    format = f"{{:>{length}s}}"
                elif format.endswith("}"):
                    length = int(format[2:-1].strip("<>s"))
                    if len(val) > length:
                        # val = val[:length]
                        val = val.strip("_ ")
            try:
                data[name] = format.format(val)
            except Exception as err:
                raise RuntimeError(f"Error formatting {name}={val} with {format=}") from err
    return data


def norm_language(language):
    # if "-" in language and len(language) == 5:
    #     return "xx-xx"
    return language


def to_dict_key(key):
    if isinstance(key, tuple):
        return "//".join([str(k) if k not in [None, ""] else "---" for k in key])
    return key


if __name__ == "__main__":
    _folder = os.path.dirname(__file__)

    import argparse

    parser = argparse.ArgumentParser(
        description="Make a csv table with datasets statistics.", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--stat_folder",
        type=str,
        default=os.path.join(_folder, "stats_raw"),
        help="Folder with statistics about raw sub-datasets, in json formats (generated by data.py)",
    )
    parser.add_argument(
        "--tokencount_folder",
        type=str,
        default=os.path.join(_folder, "stats_tokens"),
        help="Folder with statistics about tokenized sub-datasets, in json formats (generated by count_tokens.py)",
    )
    parser.add_argument(
        "--output_metadata_file",
        type=str,
        default=os.path.join(_folder, "stats_datasets.csv"),
        help="Output file with statistics about datasets",
    )
    parser.add_argument(
        "--output_metadata_file_detailed",
        type=str,
        default=os.path.join(_folder, "stats_datasets_detailed.csv"),
        help="Output file with detailed statistics about datasets",
    )
    args = parser.parse_args()

    tokencount_folder = args.tokencount_folder
    stat_folder = args.stat_folder
    output_metadata_file = args.output_metadata_file
    output_metadata_file_detailed = args.output_metadata_file_detailed
    avoid_subsets = True

    global total_tokens
    ground_total_tokens = 0
    total_tokens = {}
    total_sequences_check = {}
    if tokencount_folder:
        assert os.path.exists(tokencount_folder)
        for fn in os.listdir(tokencount_folder):
            if not fn.endswith(".json"):
                continue
            if ignore_datasets(fn):
                continue
            data_fullname = os.path.join(tokencount_folder, fn)
            data = json.load(open(data_fullname, encoding="utf8"))

            language, name, subset = to_language_name_subset(fn)

            ground_total_tokens += data["total_tokens"]
            keys = [(language, name, subset)]
            if subset not in ["---", "", None]:
                keys.append((language, name, "---"))
            if name not in ["---", "", None]:
                keys.append((language, "---", "---"))
            if language not in ["---", "", None]:
                keys.append(("---", "---", "---"))
            for key in keys:
                key = to_dict_key(key)
                total_tokens[key] = total_tokens.get(key, 0) + data["total_tokens"]
                total_sequences_check[key] = total_sequences_check.get(key, 0) + data["total_sequences"]

        # Sort dictionary
        total_tokens = dict(sorted(total_tokens.items(), key=lambda item: item[0]))
        total_sequences_check = dict(sorted(total_sequences_check.items(), key=lambda item: item[0]))

    for ONLY_DETAILED in (
        False,
        True,
    ):
        rows = []
        rows_detailed = []
        has_details = {}
        for fn in sorted(os.listdir(stat_folder)):
            if not fn.endswith(".json"):
                continue
            if not fn.startswith("stats_"):
                continue
            if ignore_datasets(fn):
                continue
            data_fullname = os.path.join(stat_folder, fn)
            data = json.load(open(data_fullname, encoding="utf8"))

            name, subset = to_name_subset(fn)

            if "pes2o" in name.lower():
                if subset in ["train", "s2orc-validation", "s2ag-validation"]:
                    continue
            if name == "CroissantAligned" and "augment" in subset:
                continue

            language, name, subset = to_language_name_subset(name, subset)

            # This is ugly...
            if avoid_subsets:
                if ONLY_DETAILED:
                    if name.lower() not in ["thestack", "pile", "otherfr"] and subset:
                        continue
            else:
                subset = None

            if language is None:
                continue
            row = {
                "language": language,
                "name": name,
            }
            if ONLY_DETAILED:
                if (
                    name.lower() in ["americanstories", "parlement", "opendata"] or name.lower().startswith("claire")
                ) and subset:
                    continue

            if subset:
                row["subset"] = subset
                has_details[name] = True
            try:
                row.update({KEYS[k]: v for k, v in data.items() if k in KEYS})
            except Exception as err:
                raise RuntimeError(f"Error processing {data_fullname}") from err
            assert SORT_BY in row, f"Missing {SORT_BY} in {row}"
            if subset:
                rows_detailed.append(row)
            else:
                rows.append(row)

        for row in rows:
            if row["name"] in has_details:
                continue
            row = row.copy()
            row["subset"] = ""
            assert SORT_BY in row, f"Missing {SORT_BY} in {row}"
            rows_detailed.append(row)

        # Make total per language
        totals_per_language = {}
        totals_per_dataset = {}
        totals = {k: 0 for k in KEYS.values()}
        for row in rows_detailed if ONLY_DETAILED else rows:
            language = norm_language(row["language"])
            ds_name = row["name"] + row["subset"] if ONLY_DETAILED else row["name"]
            if language not in totals_per_language:
                totals_per_language[language] = {k: 0 for k in KEYS.values()}
                totals_per_language[language]["#datasets"] = 0
            totals_per_language[language]["#datasets"] += 1
            assert ds_name not in totals_per_dataset, f"Duplicate {ds_name}"
            totals_per_dataset[ds_name] = {}
            for k, v in row.items():
                if k in KEYS.values():
                    totals_per_language[language][k] += v
                    totals_per_dataset[ds_name][k] = v
                    totals[k] += v

        # Add total rows
        totals = {
            "language": "-" * 3,
            "name": "-" * 3,
        } | totals
        rows.append(row)
        totals = totals.copy()
        totals["subset"] = "-" * 3
        rows_detailed.append(totals)
        totals_per_dataset["-" * 3] = totals
        for language, data in totals_per_language.items():
            row = {
                "language": language,
                "name": "-" * 3,  # TOTAL
            } | {k: v for k, v in data.items() if k in KEYS.values()}
            assert SORT_BY in row, f"Missing {SORT_BY} in {row}"
            if data["#datasets"] > 0:  # 1
                rows.append(row)
                row = row.copy()
                row["subset"] = "-" * 3  # TOTAL
                rows_detailed.append(row.copy())

        def sort_function(row):
            return (
                (8 if row["language"] == "---" else 0)
                + (4 if row["name"] == "---" else 0)
                + (2 if row.get("subset") == "---" else 0),
                {"code": 0, "fr": 1e32}.get(
                    row["language"], totals_per_language.get(norm_language(row["language"]), {SORT_BY: 1e20})[SORT_BY]
                ),
                row[SORT_BY],
                row["name"],
            )

        rows = sorted(
            rows,
            key=lambda row: sort_function(row),
            reverse=True,
        )
        rows_detailed = sorted(
            rows_detailed,
            key=lambda row: sort_function(row),
            reverse=True,
        )

        fieldnames = ["language", "name", "subset"] + get_stat_names(bool(tokencount_folder))
        header_with_spaces = format_stats_display(dict(zip(fieldnames, fieldnames)), ONLY_DETAILED)

        if ONLY_DETAILED:
            output_metadata_file_detailed = output_metadata_file
        else:
            with open(output_metadata_file, "w", encoding="utf8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
                # writer.writeheader()
                writer.writerow(header_with_spaces)
                for row in rows:
                    row = compute_extra_stats(row, tokencount_folder)
                    row = format_stats_display(row, ONLY_DETAILED)
                    writer.writerow(row)

        with open(output_metadata_file_detailed, "w", encoding="utf8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
            writer.writerow(header_with_spaces)
            for row in rows_detailed:
                row = compute_extra_stats(row, tokencount_folder)
                row = format_stats_display(row, ONLY_DETAILED)
                writer.writerow(row)
