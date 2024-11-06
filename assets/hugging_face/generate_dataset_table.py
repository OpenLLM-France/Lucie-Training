import csv
import os
import re

import mistune
import pandas as pd
import slugify

_parent_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
_stats_filename = os.path.join(_parent_folder, "assets", "stats_datasets.csv")
_stats_filename_detailed = os.path.join(_parent_folder, "assets", "stats_datasets_detailed.csv")

_web_datasets = ["RedPajama", "FineWebEdu"]

_meta_datasets = [
    "Pile",
    "OtherFr",
    "Wikiother",
    # "OpenData",
]

# List of datasets for which the detailed statistics are taken from the detailed CSV file
_detail_datasets = [
    "Claire",
    "Wikiother",
    "wikisource",
    "wiktionary",  # Ugly
    # "AmericanStories",
]

_shown_fields = [
    "M docs",
    "B words",
    "B tokens",
    "B chars",
]

_show_fields_in_details = [
    "B words",
]
_sorting_field = _show_fields_in_details[0]  # "B chars"


def load_stats():
    def norm_field(key, val, trace_error=None):
        key = key.strip()
        val = val.strip()
        try:
            if "#" in key:
                val = float(val)
            elif key.startswith("M "):
                val = float(val)  # * 1_000_000
            elif key.startswith("K "):
                val = float(val)  # * 1_000
            elif key.startswith("B "):
                val = float(val)  # * 1_000_000_000
        except ValueError:
            return key, None
            # raise ValueError(f"Error parsing '{key}' : '{val}' ({trace_error})") from err
        return key, val

    def compile_stats(row):
        if row["language"] == "code":
            row["language"] = f"{row['subset']}".upper()

        category = row["category"]
        category = {
            "legi_written": "Legislative Texts",
            "legi_spoken": "Legislative Transcripts",
            "legi_dialogue": "Legislative Transcripts",
            "aligned": "Parallel Corpora",
        }.get(category, category)
        category = " ".join([w.capitalize() for w in category.replace("_", " ").split()])
        row["category"] = category

        subset = row["subset"]
        subset = {
            "questions_ecrites_parlement": "QuestionsEcritesParlement",
            "interventions_parlement": "InterventionsParlement",
            "LEGI": "LEGI",
            "amendements_parlement": "AmendementsParlement",
        }.get(subset, subset)
        row["subset"] = subset

        name = row["name"]
        for ds in _web_datasets:
            if name.startswith(ds):
                name = name[: len(ds)]
                break
        for ds in _meta_datasets:
            if name.startswith(ds):
                name = row.get("subset", "")
                if not name:
                    name = ds
                assert name, f"Missing subset for {row['name']}"
                if "other" not in ds.lower():
                    name = f"{ds} ({name})"
                break
        if name.endswith("." + row["language"]):
            name = name[: -(1 + len(row["language"]))]
        name = {
            "ValidatedYouTube": "YouTube",
        }.get(name, name)
        row["name"] = name

        if "extra" not in row:
            row["extra"] = {}

        return row

    def merge_stats(row1, row2, orig_name):
        extra = row1.get("extra", {})
        merged = row1.copy()
        subset = row2.get("subset")
        sort_by_count = True
        if not subset:
            for ds in _web_datasets:
                if orig_name.startswith(ds):
                    subset = orig_name.split("-")[-1]
                    sort_by_count = False
                    break
        if subset:
            info = f"**{subset}**"
            if round(row2[_show_fields_in_details[0]], 3) > 0:
                info += " ("
                info += ", ".join([f"{row2[k]} {k}" for k in _show_fields_in_details])
                info += ")"
            sort_criterion = -row2[_sorting_field] if sort_by_count else subset
            extra[subset] = (sort_criterion, info)
        for k, v in row2.items():
            assert k in merged
            if isinstance(v, (float, int)):
                merged[k] += v
            elif isinstance(v, str):
                if v != row1[k]:
                    assert k in ["subset"], f"'{k}' : {v} != {row1[k]} for {merged.get('name', row2['name'])}"
                    merged[k] = ""  # f"{row1[k]} / {v}"
        if extra:
            merged["extra"] = extra
        return merged

    data = {}
    for stat_filename, only_names, excluded_names in [
        (_stats_filename, None, _detail_datasets),
        (_stats_filename_detailed, _detail_datasets, None),
    ]:
        with open(stat_filename) as f:
            reader = csv.DictReader(f)
            for row in reader:
                row = dict(norm_field(k, v, row) for k, v in row.items())
                if "---" in row.values():
                    continue
                orig_name = row["name"]
                row = compile_stats(row)
                name = row["name"]
                # NOCOMMIT
                # if "Wikiother" in orig_name:
                #     if only_names:
                #         print(f"{name=} {only_names=} => {name not in only_names}")
                #     if excluded_names:
                #         print(f"{name=} {excluded_names=} => {name in excluded_names}")
                #     import pdb; pdb.set_trace()
                if only_names and name not in only_names:
                    continue
                if excluded_names and name in excluded_names:
                    continue
                key = tuple(row[k] for k in ["name", "language"])

                if key in data:
                    # Merge stats
                    row = merge_stats(data[key], row, orig_name)

                data[key] = row

    df = pd.DataFrame(data.values())
    df["extra"] = df["extra"].apply(
        lambda extra: ", ".join([info for _, info in sorted(extra.values())]) if len(extra) > 1 else ""
    )

    return df


def write_md_table_row(fields, row=None, header=None):
    if row is None:
        return "\n".join(
            write_md_table_row(fields, row, header=header)
            for row, header in [
                ({f: to_header(f) for f in fields}, True),
                ({f: "-" * 1 for f in fields}, False),
            ]
        )
    cells = [to_str(f, row[f], header=header) for f in fields]
    return "| " + " | ".join(cells) + " |"


def to_str(f, x, header=None):
    if isinstance(x, str):
        internal_link = to_link(x)

    if isinstance(x, float):
        # Not too many decimals
        return str(round(x, 3))
    elif f == x and x == "extra":
        x = ""
        # bold
    elif isinstance(x, str) and (header is True or (header is None and f in ["name", "language"])):
        x = f"**{x}**"

    # Add internal link
    if isinstance(x, str):
        if f == "name" and x not in ["name", "-"]:
            # x=f"<a href=\"#{internal_link}\">{x}</a>"
            x = f"[{x}](#{internal_link})"

    return str(x)


def to_link(x):
    x = {
        "RedPajama": "RedPajama (v2)",
        "Claire": "Claire (French and English)",
    }.get(x, x)
    if x.startswith("Pile"):
        x = "Pile (Uncopyrighted)"
    if x.startswith("Wiki"):
        x = "Wikipedia, Wikisource, Wiktionary"
    if x.startswith("Europarl"):
        x = "Europarl (monolingual and parallel)"
    return slugify.slugify(x)


def to_header(x):
    if x == "name":
        x = "subset"
    return x


def convert_markdown_table_to_html(
    markdown,
    html_doc,
    headers=None,
    # center_title= True,
):
    generated_html = mistune.html(markdown)

    # Fix multi-rows
    for num_columns in list(range(10, 1, -1)):
        regex_to = rf'\1<td colspan="{num_columns}" style="text-align: center;"><u>\2</u></td></tr>'
        # if not center_title:
        #     regex_to = rf'\1<td colspan="{num_columns-5}"></td><td colspan="{num_columns-2}"><u>\2</u></td></tr>'
        if headers:
            headers = headers.strip("<>/")

            def regex_to(match):
                title = match.group(2)
                title = re.sub("<[^>]*>", "", title)
                title = f"<{headers} id={slugify.slugify(title)}>{title}</{headers}>"
                return f'{match.group(1)}<td colspan="{num_columns}">{title}</td></tr>'

        generated_html = re.sub(
            rf"(<tr>\s*)<td>(.+)</td>\s*(<td>\s*</td>\s*){{{num_columns-1}}}</tr>", regex_to, generated_html
        )

    html_doc.write(generated_html)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_md",
        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "README_dataset_table.md"),
        nargs="?",
    )
    parser.add_argument(
        "output_html",
        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "README_dataset_table.html"),
        nargs="?",
    )
    parser.add_argument(
        "--headers",
        default="h4",
        help="HTML headers for the table (if None, it will just be centered text)",
    )
    args = parser.parse_args()

    df = load_stats()
    print(df)

    categories = df["category"].unique()
    print(categories)

    fields = (
        [
            "name",
            "language",
        ]
        + _shown_fields
        + [
            "extra",
        ]
    )

    def key_row(row, df):
        num_tokens = row["B tokens"]
        num_tokens_total = df[df["name"] == row["name"]]["B tokens"].sum()
        return (-num_tokens_total, -num_tokens, row["name"])

    def key_category(category, df):
        if "code" in category.lower() or "programming" in category.lower():
            return 0
        num_tokens = df[df["category"] == category]["B tokens"].sum()
        return -num_tokens

    if args.output_md:
        with open(args.output_md, "w") as f:
            f.write(write_md_table_row(fields) + "\n")
            for category in sorted(categories, key=lambda x: key_category(x, df)):
                f.write(f"| ***{category}*** " + ("|" * len(fields)) + "\n")
                df_cat = df[df["category"] == category]
                rows = [row for irow, row in df_cat.iterrows()]
                for row in sorted(rows, key=lambda x: key_row(x, df_cat)):
                    f.write(write_md_table_row(fields, row) + "\n")

    if args.output_html:
        assert args.output_md, "Need to generate the markdown table first"
        with open(args.output_md) as f_in:
            table_content = f_in.readlines()

        with open(args.output_html, "w") as f_out:
            convert_markdown_table_to_html(
                "".join(table_content),
                f_out,
                # headers="h4",
                headers=args.headers,
            )
