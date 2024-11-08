import csv
import os
import re

import bs4
import matplotlib.pyplot as plt
import mistune
import numpy as np
import pandas as pd
import slugify

_parent_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
_stats_filename = os.path.join(_parent_folder, "assets", "stats_datasets.csv")
_stats_filename_detailed = os.path.join(_parent_folder, "assets", "stats_datasets_detailed.csv")

USE_HATCH_FOR_CATEGORIES = True

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

_sorting_field = "B tokens"  # "B chars" # "B words"


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


def conform_extra(subset, row, sort_by_count=True, name=None):
    count = row[_sorting_field]
    info = subset
    try:
        subset_int = int(subset)
        info = subset_int
    except ValueError:
        subset_int = None
    sort_criterion = -row[_sorting_field] if sort_by_count else (subset if subset_int is None else -subset_int)
    return (name, sort_criterion, count, info)


def figure_name(name, graph_type="histogram", suffix=""):
    return (f"Composition of {name}{suffix}", f"figures/fig_distribution_{slugify.slugify(name)}_{graph_type}.png")


def plot_extra_distribution(extra, name=None):
    assert len(extra)
    extra_list = sorted(extra.values())
    sum_count = sum(count for _, _, count, _ in extra_list)
    ratios = [count * 100.0 / sum_count for _, _, count, _ in extra_list]
    name = name or extra_list[0][0]
    is_numeric = isinstance(extra_list[0][-1], int)
    if is_numeric:
        graph_type = "histogram"
        suffix = " by year"
    else:
        graph_type = "pie"
        suffix = ""
    descr, figname = figure_name(name, graph_type=graph_type, suffix=suffix)

    # Clear figure
    plt.clf()
    if graph_type == "histogram":
        # Histogram of years
        plt.bar([val for _, _, _, val in extra_list], ratios, width=1)
        plt.ylabel("Percentage")
        if suffix:
            plt.xlabel(suffix.split()[-1])
        # Avoid non-integer xticks
        (xvalues, xlabels) = plt.xticks()
        plt.xticks([v for v in xvalues if int(v) == v])
    else:
        # Pie chart
        labels = [val for _, _, _, val in extra_list]
        do_not_plot_small = len(labels) > 3
        if do_not_plot_small:
            for i, percent in enumerate(ratios):
                if percent < 1:
                    labels[i] = ""
        plt.pie(
            ratios,
            labels=labels,
            autopct=format_percentage if do_not_plot_small else "%1.1f%%",
            startangle=90 * 2,
            counterclock=False,
        )
    if name:
        plt.title(descr)
        plt.savefig(figname, bbox_inches="tight")
    return descr, figname


def format_extra_in_table(extra, use_figures=True):
    hack_am = extra == "HACK_AmericanStories"
    descr, figname = None, None
    if hack_am:
        name = "AmericanStories (English)"
        # This one is not generated every time ... (cause we don't have number of tokens for each year)
        descr, figname = figure_name(name, "histogram", suffix=" by year")
    elif len(extra) > 1:
        descr, figname = plot_extra_distribution(extra)
    md_string = f"[composition details]({figname})"  # f"[{descr}]({figname})"

    if len(extra) > 1 or hack_am:
        if use_figures and figname:
            return md_string
        sum_count = sum(count for _, _, count, _ in extra.values())
        return ", ".join(
            [
                f"**{name}** ({precision_at_least(count*100./sum_count, 1)} %)"
                for _, _, count, name in sorted(extra.values())
            ]
        )
    if not use_figures and len(extra) == 1 and list(extra.keys())[0].startswith("EuroparlAligned"):
        name = list(extra.values())[0][2]
        if name:
            return f"**{name}**"
    return ""


def merge_stats(row1, row2, orig_name):
    extra = row1.get("extra", {})
    merged = row1.copy()
    sort_by_count = True
    for k, v in row2.items():
        assert k in merged
        if k == "extra":
            continue
        if isinstance(v, (float, int)):
            merged[k] += v
        elif isinstance(v, str):
            if v != row1[k]:
                assert k in ["subset"], f"'{k}' : {v} != {row1[k]} for {merged.get('name', row2['name'])}"
                merged[k] = ""  # f"{row1[k]} / {v}"
    name = merged.get("name", row2["name"])
    language = merged.get("language", row2["language"])
    if language:
        name += f" ({format_language(language, include_lang_code=False)})"
    for ds in _web_datasets:
        if orig_name.startswith(ds):
            sort_by_count = False
            break
    if row2.get("subset"):
        for row in row1, row2:
            subset = row.get("subset")
            if not subset:
                continue
            language = row.get("language")
            assert language, f"Missing language for {row['name']}"
            extra[(subset, language)] = conform_extra(subset, row, sort_by_count=sort_by_count, name=name)
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
    sum_tokens = sum(row["B tokens"] if row.get("B tokens") else 1 for row in data.values())
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
        lang: sum(
            row["B tokens"] if row.get("B tokens") else 1
            for row in data.values()
            if to_generic_language(row["language"]) == lang
        )
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
                lang if language == "code" else subset: conform_extra(
                    lang if language == "code" else subset, row, name=format_language(language, include_lang_code=False)
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

    df.loc[df["name"] == "AmericanStories", "extra"] = "HACK_AmericanStories"
    df["extra"] = df["extra"].apply(format_extra_in_table)

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
    if np.isnan(x):
        return "0"
    if x < 10**-7:
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
        # "code": "Programming Languages",
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


# format_percentage = "%1.1f%%"
def format_percentage(pct):
    return f"{pct:.1f}%" if pct > 1 else ""


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

    plt.figure()  # For the pie charts, etc.
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
            return 1
        if "multilingual" in category.lower():
            return 0
        num_tokens = df[df["category"] == category]["B tokens"].sum()
        if category.startswith("Legislative"):
            num_tokens = sum(
                df[df["category"] == subcat]["B tokens"].sum()
                for subcat in ["Legislative Texts", "Legislative Transcripts", "Legislative"]
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

        map_hatch = {
            "fr": "/",
            "en": "*",
            "de": ".",
            "es": "-",
            "it": "|",
            "code": "o",
            "parallel": "+",
            "Multilingual Parallel Corpora": "",
            "Programming": "",
            "Web": "/",
            "Newspaper": ".",
            "Technical": "x",
            "Book": "*",
            "Legislative": "-",
            "Legislative Texts": "-",
            "Legislative Transcripts": "-",
            "Wiki": "o",
            "Math": "\\",
            "Forum": "|",
            "Dialogue": "+",
        }

        map_colors = {
            "fr": "royalblue",  # "mediumblue", # "blue",
            "en": "crimson",  # "red",
            "de": "mediumpurple",  # "purple",
            "es": "gold",  # "orange",
            "it": "seagreen",  # "olive", # "green",
            "code": "black",  # "gray",
            "parallel": "gray",  # "pink",
        }

        min_percent_for_label = 0.8

        categories = [c for c in categories if c]
        categories = sorted(categories, key=lambda x: key_category(x, df))
        for i, c in enumerate(categories):
            if c.startswith("Legislative"):
                categories[i] = "Legislative"
        categories = list(dict.fromkeys(categories))  # Remove duplicates without resorting
        num_colors = len(categories)
        rainbow_colors = [plt.cm.gist_rainbow(i / num_colors) for i in range(num_colors)]
        languages = [k for k in map_colors.keys() if len(k) == 2]
        # rainbow_colors_1 = rainbow_colors[: len(rainbow_colors) // 2]
        # rainbow_colors_2 = rainbow_colors[len(rainbow_colors) // 2 :]
        # rainbow_colors = [color for pair in zip(rainbow_colors_1, rainbow_colors_2) for color in pair]

        for STAT_NAME in (_sorting_field,):  # ("B tokens", "B words", "B chars", ):
            pie_values = []
            pie_labels = []
            pie_colors = []
            pie_hatches = []
            pie_categories = []
            pie_languages = []

            legend_categories = {}
            legend_languages = {}
            count_per_category = {}
            count_per_language = {}
            for _, row in df.iterrows():
                category = row["category"]
                if category.startswith("Legislative"):
                    category = "Legislative"
                language = to_generic_language(row["language"], parallel=True)
                subset = row["name"]
                if language == "code":
                    subset = row["language"]
                if not category or not subset or not language or subset.lower() in ["total", "name"]:
                    continue
                if subset == "RedPajama":
                    subset += f"-{row['language']}"
                pie_values.append(row[STAT_NAME])
                pie_categories.append(category)
                pie_languages.append(language)
                pie_labels.append(subset)
                pie_colors.append(
                    map_colors.get(language, "black")
                    if USE_HATCH_FOR_CATEGORIES
                    else rainbow_colors[categories.index(category)]
                )
                pie_hatches.append(map_hatch.get(category if USE_HATCH_FOR_CATEGORIES else language, "") * 2)
                count_per_category[category] = count_per_category.get(category, 0) + row[STAT_NAME]
                count_per_language[language] = count_per_language.get(language, 0) + row[STAT_NAME]
                rec_hatch = plt.Rectangle((0, 0), 1, 1, fc="white", hatch=pie_hatches[-1], edgecolor="black")
                rec_color = plt.Rectangle((0, 0), 1, 1, fc=pie_colors[-1], hatch="", edgecolor="black")
                rec_hatch_and_color = plt.Rectangle(
                    (0, 0), 1, 1, fc=pie_colors[-1], hatch=pie_hatches[-1], edgecolor="black"
                )
                if len(language) == 2:
                    legend_categories[category] = rec_hatch
                    legend_languages[language] = rec_color
                else:
                    legend_categories[category] = rec_hatch_and_color

            new_labels = []
            for lab, v in zip(pie_labels, pie_values):
                percentage = v / sum(pie_values) * 100
                sep = "\n" if percentage > 2.6 else " "
                label = f"$\\bf{{{lab}}}${sep}({percentage:.1f}%)"
                new_labels.append(label)
            pie_labels = new_labels

            pie_data = list(zip(pie_values, pie_labels, pie_colors, pie_hatches, pie_categories, pie_languages))

            def get_counts(x):
                value_subset = x[0]
                value_dataset = sum(p[0] for p in pie_data if p[1] == x[1])
                value_category = sum(p[0] for p in pie_data if p[2] == x[2])
                value_language = sum(p[0] for p in pie_data if p[3] == x[3])
                # if x[3] == map_hatch["code"]:
                #     value_category = 0
                # if x[3] == map_hatch["parallel"]:
                #     value_category = 1
                value_category = -categories.index(x[4])
                value_language = -list(map_hatch.keys()).index(x[5])
                return (value_category, value_language, value_dataset, value_subset)

            pie_data = sorted(pie_data, key=get_counts, reverse=True)

            pie_values, pie_labels, pie_colors, pie_hatches, pie_categories, pie_languages = zip(*pie_data)
            pie_labels = list(pie_labels)
            pie_hatches = list(pie_hatches)

            sum_values = sum(pie_values)
            pie_values = [v / sum_values * 100 if sum_values else 0 for v in pie_values]
            for i, v in enumerate(pie_values):
                if v < min_percent_for_label:
                    pie_labels[i] = ""  # "other"
                    # pie_hatches[i] = pie_hatches[i] * 2
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
                explode=[0.1 if lab else 0.05 for lab in pie_labels],
            )
            plt.axis("equal")

            # Custom legend
            def format_category_for_pie(category):
                cat = category.replace(" Parallel Corpora", "")
                percent = count_per_category[category] / sum(count_per_category.values()) * 100
                return f"{cat} ({precision_at_least(percent, 0)}%)"

            def format_language_for_pie(language):
                lang = format_language(language, include_lang_code=False)
                percent = count_per_language[language] / sum(count_per_language.values()) * 100
                return f"{lang} ({precision_at_least(percent, 0)}%)"

            nothing = plt.Rectangle((0, 0), 1, 1, fc="white", edgecolor="white")
            legend_and_labels = (
                [(nothing, "$\\bf{Categories}$")]
                + [
                    (legend_categories[label], format_category_for_pie(label))
                    for label in categories
                    if label in legend_categories
                ]
                + [(nothing, "$\\bf{Languages}$")]
                + [
                    (legend_languages[label], format_language_for_pie(label))
                    for label in languages
                    if label in legend_languages
                ]
                + [(nothing, "") for i in range(len(categories) - len(languages))]
            )
            legend, labels = zip(*legend_and_labels)
            plt.legend(
                legend,
                labels,
                ncols=2,
                fontsize="large",
                # markerscale=20,
                shadow=True,
                # loc="upper left",
                loc="best",
                bbox_to_anchor=(-0.1, 0.0, 0.5, 1),
            )
            # plt.title(STAT_NAME)

        plt.show()
