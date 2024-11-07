import csv
import os
import re

import bs4
import matplotlib.pyplot as plt
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
    "B tokens",  # "B words",
]
_sorting_field = _show_fields_in_details[0]  # "B chars"


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
        "aligned": "Multilingual Parallel Corpora",
    }.get(category.lower(), category)
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
            subset = name.split("-")[-1]
            name = name[: len(ds)]
            row["subset"] = subset
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


def format_extra_info_string(subset, row, sort_by_count=True):
    assert len(_show_fields_in_details) == 1
    count = row[_show_fields_in_details[0]]
    info = f"**{subset}**"
    # if True:  # round(row[_show_fields_in_details[0]], 3) > 0:
    #     info += " ("
    #     info += ", ".join([f"{precision_at_least(row[k], 1)} {k}" for k in _show_fields_in_details])
    #     info += ")"
    try:
        subset_int = int(subset)
    except ValueError:
        subset_int = None
    sort_criterion = -row[_sorting_field] if sort_by_count else (subset if subset_int is None else -subset_int)
    return (sort_criterion, count, info)


def merge_stats(row1, row2, orig_name):
    extra = row1.get("extra", {})
    merged = row1.copy()
    sort_by_count = True
    for ds in _web_datasets:
        if orig_name.startswith(ds):
            sort_by_count = False
            break
    if row2.get("subset"):
        for row in row1, row2:
            subset = row.get("subset")
            if not subset:
                continue
            extra[subset] = format_extra_info_string(subset, row, sort_by_count)
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


def to_generic_language(lang, parallel=False):
    if lang.isupper():
        return "code"
    if "-" in lang and parallel:
        return "parallel"
    if not lang:
        return "(any)"
    return lang


def postprocess_extra(extra):
    if len(extra) > 1:
        sum_count = sum(count for _, count, _ in extra.values())
        return ", ".join(
            [f"{name} ({precision_at_least(count*100./sum_count, 1)} %)" for _, count, name in sorted(extra.values())]
        )
    if len(extra) == 1 and list(extra.keys())[0].startswith("EuroparlAligned"):
        return list(extra.values())[0][2]
    return ""


def load_stats():
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
                if name in ["Persee"]:
                    # Exclude from released data
                    continue
                if only_names and name not in only_names:
                    continue
                if excluded_names and name in excluded_names:
                    continue
                key = tuple(row[k] for k in ["name", "language"])

                if key in data:
                    # Merge stats
                    row = merge_stats(data[key], row, orig_name)

                data[key] = row

    # Add totals
    sum_docs = sum(row["M docs"] for row in data.values())
    sum_words = sum(row["B words"] for row in data.values())
    sum_tokens = sum(row["B tokens"] for row in data.values())
    sum_chars = sum(row["B chars"] for row in data.values())

    languages = {to_generic_language(row["language"]) for row in data.values()}
    sum_docs_per_lang = {
        lang: sum(row["M docs"] for row in data.values() if to_generic_language(row["language"]) == lang)
        for lang in languages
    }
    sum_words_per_lang = {
        lang: sum(row["B words"] for row in data.values() if to_generic_language(row["language"]) == lang)
        for lang in languages
    }
    sum_tokens_per_lang = {
        lang: sum(row["B tokens"] for row in data.values() if to_generic_language(row["language"]) == lang)
        for lang in languages
    }
    sum_chars_per_lang = {
        lang: sum(row["B chars"] for row in data.values() if to_generic_language(row["language"]) == lang)
        for lang in languages
    }
    for language in sorted(languages):
        data[("", language)] = {
            "name": "TOTAL",
            "language": language,
            "category": "",
            "M docs": sum_docs_per_lang[language],
            "B words": sum_words_per_lang[language],
            "B tokens": sum_tokens_per_lang[language],
            "B chars": sum_chars_per_lang[language],
            "extra": {
                lang if language == "code" else subset: format_extra_info_string(
                    lang if language == "code" else subset, row
                )
                for (subset, lang), row in data.items()
                if to_generic_language(row["language"]) == language
            },
        }
    data[("", "")] = {
        "name": "TOTAL",
        "language": "",
        "category": "",
        "M docs": sum_docs,
        "B words": sum_words,
        "B tokens": sum_tokens,
        "B chars": sum_chars,
        "extra": {},
    }

    df = pd.DataFrame(data.values())

    df["extra"] = df["extra"].apply(postprocess_extra)

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
    cells = [format_str(f, row[f], header=header) for f in fields]
    return "| " + " | ".join(cells) + " |"


def format_str(f, x, header=None):
    x_orig = x
    if isinstance(x, str):
        x = x.strip()
        if not x:
            return ""
        internal_link = to_link(x)

    if f == "language":
        x = format_language(x, include_lang_code=True)

    if isinstance(x, float):
        # Not too many decimals
        return precision_at_least(x)
    elif f == x and x == "extra":
        x = ""
        # bold
    elif isinstance(x, str) and (header is True or (header is None and f in ["name", "language"])):
        x = f"**{x}**"

    # Add internal link
    if isinstance(x, str):
        if f == "name" and x and x_orig.lower() not in ["name", "-", "total"]:
            # x=f"<a href=\"#{internal_link}\">{x}</a>"
            x = f"[{x}](#{internal_link})"

    return str(x)


def precision_at_least(x, prec=3, length=""):
    if x == 0:
        return f"{0:{length}.{prec}f}"
    if round(x, prec) >= 100 * (10**-prec):
        return f"{x:{length}.{prec}f}"
    return precision_at_least(x, prec + 1)


def to_link(x):
    x = {
        "RedPajama": "RedPajama (v2)",
        "Claire": "Claire (French and English)",
    }.get(x, x)
    if x.startswith("Pile"):
        x = "Pile (Uncopyrighted)"
    if x.lower().startswith("wik"):
        x = "Wikipedia, Wikisource, Wiktionary"
    if x.startswith("Europarl"):
        x = "Europarl (monolingual and parallel)"
    return slugify.slugify(x)


def to_header(x):
    if x == "name":
        x = "subset"
    return x


def format_language(lang_code, include_lang_code=True):
    lang = {
        "fr": "French",
        "en": "English",
        "de": "German",
        "es": "Spanish",
        "it": "Italian",
        "code": "Programming Languages",
        "parallel": "Multilingual Parallel",
    }.get(lang_code, lang_code)
    if include_lang_code and lang != lang_code:
        lang = f"{lang} ({lang_code})"
    return lang


def convert_markdown_table_to_html(
    markdown,
    html_doc,
    headers=None,
    # center_title= True,
):
    generated_html = mistune.html(markdown)

    # Fix multi-Columns
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

    # Add multi-rows when cell name is the same
    generated_html = add_rowspan_to_table(generated_html)

    html_doc.write(generated_html)


def add_rowspan_to_table(html):
    soup = bs4.BeautifulSoup(html, "html.parser")
    table = soup.find("table")

    rows = table.find_all("tr")
    if not rows:
        return html

    first_column_cells = [row.find_all("td")[0] if row.find_all("td") else None for row in rows[1:]]  # Skip header row
    rowspan_map = {}

    for i, cell in enumerate(first_column_cells):
        if cell:
            cell_text = cell.get_text()
            if cell_text in rowspan_map:
                rowspan_map[cell_text]["count"] += 1
                rowspan_map[cell_text]["rows"].append(i + 1)
            else:
                rowspan_map[cell_text] = {"count": 1, "rows": [i + 1]}

    for _, data in rowspan_map.items():
        if data["count"] > 1:
            first_row_index = data["rows"][0]
            first_row_cell = rows[first_row_index].find_all("td")[0]
            first_row_cell["rowspan"] = data["count"]
            first_row_cell["style"] = "vertical-align: top;"  # Add CSS for vertical alignment
            for row_index in data["rows"][1:]:
                rows[row_index].find_all("td")[0].decompose()

    return str(soup)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_md",
        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "README_dataset_table.md"),
        help="Output markdown file",
        nargs="?",
    )
    parser.add_argument(
        "output_html",
        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "README_dataset_table.html"),
        help="Output HTML file",
        nargs="?",
    )
    parser.add_argument(
        "output_main",
        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "README_dataset.md"),
        help="Output main markdown file (with the HTML table included in it)",
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
        if category.startswith("Legislative"):
            num_tokens = sum(
                df[df["category"] == subcat]["B tokens"].sum()
                for subcat in ["Legislative Texts", "Legislative Transcripts"]
            )
        else:
            num_tokens = df[df["category"] == category]["B tokens"].sum()
        return -num_tokens

    if args.output_md:
        with open(args.output_md, "w") as f:
            f.write(write_md_table_row(fields) + "\n")
            for category in sorted(categories, key=lambda x: key_category(x, df)):
                category_str = category
                # if not category:
                #     category_str = "All"
                if category_str:
                    f.write(f"| ***Category: {category_str}*** " + ("|" * len(fields)) + "\n")
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

    if args.output_main:
        assert args.output_html, "Need to generate the HTML table first"
        assert os.path.isfile(args.output_main), f"File '{args.output_main}' does not exist"
        with open(args.output_main) as f_in:
            main_content = f_in.read()

        tag_start = "\n<!-- TABLE START -->\n"
        tag_end = "\n<!-- TABLE END -->\n"

        table_start = re.search(tag_start, main_content).end()
        table_end = re.search(tag_end, main_content).start()

        new_content = main_content[:table_start] + open(args.output_html).read().strip() + main_content[table_end:]
        with open(args.output_main, "w") as f_out:
            f_out.write(new_content)

    if True:
        # Plot pie

        def more_dense_hatch(hatch):
            if hatch in ["o"]:
                return hatch
            if hatch in ["-", "+"]:
                return hatch * 3
            return hatch * 2

        def format_percentage(pct):
            return f"{pct:.1f}%" if pct > min_percent_for_label else ""

        hatch_styles = {
            "fr": "/",
            "en": "*",
            "de": ".",
            "es": "-",
            "it": "|",
            "code": "o",
            "parallel": "+",
        }

        min_percent_for_label = 0.6

        categories = [c for c in categories if c]
        categories = sorted(categories, key=lambda x: key_category(x, df))
        num_colors = len(categories)
        rainbow_colors = [plt.cm.gist_rainbow(i / num_colors) for i in range(num_colors)]
        # rainbow_colors_1 = rainbow_colors[: len(rainbow_colors) // 2]
        # rainbow_colors_2 = rainbow_colors[len(rainbow_colors) // 2 :]
        # rainbow_colors = [color for pair in zip(rainbow_colors_1, rainbow_colors_2) for color in pair]

        for STAT_NAME in ("B tokens",):  # "B words", "B chars":
            pie_values = []
            pie_labels = []
            pie_colors = []
            pie_hatches = []
            for _, row in df.iterrows():
                category = row["category"]
                language = to_generic_language(row["language"], parallel=True)
                subset = row["name"]
                if language == "code":
                    subset = row["language"]
                if not category or not subset or not language or subset.lower() in ["total", "name"]:
                    continue
                if subset == "RedPajama":
                    subset += f"-{row['language']}"
                pie_values.append(row[STAT_NAME])
                pie_labels.append(subset)
                pie_colors.append(rainbow_colors[categories.index(category)])
                pie_hatches.append(hatch_styles.get(language, ""))

            new_labels = []
            for lab, v in zip(pie_labels, pie_values):
                percentage = v / sum(pie_values) * 100
                sep = "\n" if percentage > 2.6 else " "
                label = f"{lab}{sep}({percentage:.1f}%)"
                new_labels.append(label)
            pie_labels = new_labels

            pie_data = list(
                zip(
                    pie_values,
                    pie_labels,
                    pie_colors,
                    pie_hatches,
                )
            )

            def get_counts(x):
                value_subset = x[0]
                value_dataset = sum(p[0] for p in pie_data if p[1] == x[1])
                value_category = sum(p[0] for p in pie_data if p[2] == x[2])
                value_language = sum(p[0] for p in pie_data if p[3] == x[3])
                # if x[3] == hatch_styles["code"]:
                #     value_category = 0
                # if x[3] == hatch_styles["parallel"]:
                #     value_category = 1
                value_category = -rainbow_colors.index(x[2])
                value_language = -list(hatch_styles.values()).index(x[3])
                return (value_category, value_language, value_dataset, value_subset)

            pie_data = sorted(pie_data, key=get_counts, reverse=True)

            pie_values, pie_labels, pie_colors, pie_hatches = zip(*pie_data)
            pie_labels = list(pie_labels)
            pie_hatches = list(pie_hatches)

            sum_values = sum(pie_values)
            pie_values = [v / sum_values * 100 for v in pie_values]
            for i, v in enumerate(pie_values):
                if v < min_percent_for_label:
                    pie_labels[i] = ""  # "other"
                    pie_hatches[i] = pie_hatches[i] * 2  # more_dense_hatch(pie_hatches[i])
            # Remove duplicates
            reference = None
            for i, v in enumerate(pie_labels):
                if reference is not None and v == pie_labels[reference]:
                    pie_labels[i] = ""
                else:
                    reference = i

            plt.figure()
            plt.pie(
                pie_values,
                labels=pie_labels,
                colors=pie_colors,
                hatch=pie_hatches,
                # autopct=format_percentage, #"%1.1f%%",
                shadow=False,
                startangle=90 * 2,
                counterclock=False,
                labeldistance=1.05,
                explode=[0.05 if lab else 0 for lab in pie_labels],
            )
            plt.axis("equal")

            # Custom legend
            labels, hatches, colors = [], [], []
            labels += categories
            hatches += ["" for i in categories]
            colors += rainbow_colors
            labels += list(hatch_styles.keys())
            hatches += list(hatch_styles.values())
            colors += ["white"] * len(hatch_styles)
            hatches[labels.index("Multilingual Parallel Corpora")] = hatch_styles["parallel"]
            hatches[labels.index("Programming")] = hatch_styles["code"]
            for label in "code", "parallel":
                i = labels.index(label)
                labels.pop(i)
                hatches.pop(i)
                colors.pop(i)
            hatches = [more_dense_hatch(c) for c in hatches]
            labels = [format_language(lab) for lab in labels]

            legend = [
                plt.Rectangle((0, 0), 1, 1, fc=colors[i], hatch=hatches[i], edgecolor="black")
                for i in range(len(labels))
            ]
            plt.legend(legend, labels, title_fontsize="large")
            plt.title(STAT_NAME)

        plt.show()
