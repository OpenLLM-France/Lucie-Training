import html
import math
import random
import urllib

import regex as re


def clean_pdf_extraction(text, html_escape=False):
    text = remove_page_numbers(text)
    text = remove_useless_lines(text)
    if html_escape:
        text = html_unescape(text)
    text = text.replace(" \n", "\n")
    text = text.strip()
    return text


def clean_pdf_extraction_and_html(text):
    return clean_pdf_extraction(text, html_escape=True)


def html_unescape(text):
    # Remove &nbsp; &amp; and other html entities
    return html.unescape(text)


def remove_useless_lines(text):
    # Remove cesures
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Remove lines when it does not start with a capital letter
    text = re.sub(r"([,;])\s*\n", r"\1 ", text)
    text = re.sub(r"\n\s*([a-z»])", r" \1", text)
    text = remove_double_spaces(text)
    return text


def remove_simple_lines(text):
    # Remove cesures
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Remove simple line breaks
    text = re.sub("(?<!\n)\n(?!\n)", " ", text)
    return text


def remove_double_spaces(text):
    return re.sub(r" {2,}", r" ", text)


def fix_legi(text):
    text = text.replace("/n", "\n")
    text = re.sub(r" *\n *", "\n", text)
    return remove_double_spaces(text)


def fix_legi_and_remove_title(text):
    text = fix_legi(text)
    lines = text.split("\n")
    if len(lines) > 1:
        return "\n".join(lines[1:])
    return text


def clean_discours(text):
    text = re.sub(r"-? ?\d+ VUES?$", "", text).rstrip()
    text = re.sub(r"[Ss]ource [^\n]$", "", text).rstrip()
    text = re.sub(r"([Ss]ource [:a-z0-9_\-, ]+)", " ", text)
    text = re.sub(r"[Ss]ource http[:a-z0-9_\-]+", " ", text)
    text = remove_double_spaces(text)
    return text


def remove_page_numbers(text, verbose=False, pattern=None):  # noqa # C901 `...` is too complex
    """
    Try to remove page numbers from a plain text version of a PDF.

    Assumptions:
    - Page numbers increase monotonically
    - Some page numbers may be missing
        (or undetected with a simple regex on OCR output).
        Maximum 7 consecutive missing page numbers.
    - Starts at page 2
    - Page numbers occurs at the beginning or at the end of a line
        (modulo some non-word characters)
    - Limited number of patterns supported :
        — 2 —, (2), [2], {2}, 2, ...
        and the same with a total like — 2/10 —, (2/10)
    """
    max_hole_length = 7
    beam = 5
    last_page_number = 0
    current_page_number = 2
    max_page_length = 80000
    current_hole_length = 0
    min_char_index = 0
    all_matches = []
    while current_hole_length < max_hole_length:
        use_page_length = len(all_matches) > 5
        if use_page_length:
            avg_length = min_char_index / (current_page_number - 1)
        candidates = []
        current_page_string = str(current_page_number)
        if set(current_page_string) > {"1"}:
            current_page_string = current_page_string.replace("1", "[1lIij]")
        current_pattern = _pattern_with_number(pattern, current_page_string) if pattern else None
        for match in re.finditer(
            rf"(\n[^\n]*\b{current_page_string}(/\d{{2,5}})?\b[^\n\w]*\n)|(\n[^\n\w]*\b{current_page_string}(/\d{{2,5}})?\b[^\n]*\n)",
            text,
        ):
            start = match.start()
            end = match.end()
            content = match.group()
            if current_pattern:
                match = re.search(current_pattern, content)
                if not match:
                    continue
                end = start + match.end()
                start += match.start()
            is_after = min_char_index < start
            is_not_too_far = start - min_char_index < max_page_length * (current_page_number - last_page_number)
            if pattern is not None or (is_after and is_not_too_far):
                if not len(candidates):
                    if use_page_length:
                        min_char_index = min(start, min_char_index + avg_length / 10)
                    else:
                        min_char_index = start
                    last_page_number = current_page_number
                if len(candidates) < beam:
                    candidates.append((start, match.group(), end))
                else:
                    break
        if len(candidates):
            all_matches.append(candidates)
            current_hole_length = 0
        else:
            current_hole_length += 1
        current_page_number += 1
        # Can stop looking for pattern after 100 pages
        if not pattern and current_page_number > 100:
            break
    if verbose:
        print(f"{len(all_matches)} pages found with pattern {pattern}")
    if pattern is None:
        counts_per_pattern = [0] * len(_page_patterns)
        for matches in all_matches:
            for _, content, _ in matches:
                for i, pattern in enumerate(_page_patterns):
                    if re.search(pattern, content):
                        counts_per_pattern[i] += 1
        argmax = max(range(len(_page_patterns)), key=lambda i: counts_per_pattern[i])
        if counts_per_pattern[argmax] == 0:
            return text
        pattern = _page_patterns[argmax]
        return remove_page_numbers(text, verbose=verbose, pattern=pattern)
    selected_matches = []
    min_char_index = 0
    for matches in all_matches:
        for start, content, end in matches:
            if start > min_char_index:
                min_char_index = end
                selected_matches.append((start, content, end))
                continue
    # Reverse the order of the matches
    selected_matches = selected_matches[::-1]
    for start, content, end in selected_matches:
        new_content = re.sub(pattern, "", content, count=1).strip()
        if new_content:
            new_content = "\n" + new_content + "\n"
        else:
            new_content = "\n"
        text = text[:start].rstrip() + new_content + text[end:].lstrip()
    return text


_page_patterns = (
    [rf"{re.escape(delimiter)}\s*\d{{1,5}}(/\d{{2,5}})?\s*{re.escape(delimiter)}" for delimiter in ["_", "-", "—"]]
    + [
        rf"{re.escape(delimiter_before)}\s*\d{{1,5}}(/\d{{2,5}})?\s*{re.escape(delimiter_after)}"
        for delimiter_before, delimiter_after in [("(", ")"), ("[", "]"), ("{", "}")]
    ]
    + [
        rf"(\b\d{{1,5}}(/\d{{2,5}})?\s*{re.escape(delimiter)}>?($| ))|(<?(^| ){re.escape(delimiter)}\s*\d{{1,5}}\b)"
        for delimiter in ["-", ",", "."]
    ]
)


def _pattern_with_number(pattern, n):
    return pattern.replace("\\d{1,5}", str(n))


def clean_wikipedia(text):
    text = plaintext_to_markdown(text)
    text = re.sub(
        r"(([^\s\$]*)(\$[_|\^]([^\$]|\{[^\$\}]+\})\$))+([^\s\$]*)",
        process_supersubscript,
        text,
    )
    return text


def process_supersubscript(match):
    s = "$" + match.group().replace("$", "") + "$"
    if s == "$M^{me}$":
        return "Mme"
    return s


def plaintext_to_markdown(text, website_main="wikipedia", linebreaks=2, add_toc=False, add_urls=False):  # noqa # C901 `...` is too complex
    """
    Convert plaintext (extracted by dump_wiki_htmp.py) to markdown.

    Args:
        text (str): Plaintext
        website_main (str): Website name (e.g. "wikipedia", "wikisource", "wiktionary")
            to use as links for headers. Or None if no link is needed.
        add_toc (bool): Add a Table Of Content section at the beginning of the markdown text.
    Returns:
        (str, str): (Markdown text, url)
    """
    lines = text.split("\n")

    assert linebreaks in [
        0,
        1,
        2,
        "0",
        "1",
        "2",
        "random",
    ], f"Invalid linebreaks: {linebreaks}"
    if linebreaks == "random":
        linebreaks = random.choice([0, 1, 2])
    else:
        linebreaks = int(linebreaks)

    linebreak_after_list = linebreaks > 0
    linebreak_after_table = linebreaks > 0

    linebreak_between_lines = linebreaks > 1
    linebreak_before_list = linebreaks > 1
    linebreak_before_header = linebreaks > 0
    linebreak_after_header = linebreaks > 1

    # Add :
    # - table headers
    # - new lines after table
    # - new lines after list
    # - new lines after lines
    new_lines = []
    was_list = False
    was_table = False
    was_header = True
    headers = []
    url = None
    toc = []
    markdown_subsections = []
    section_urls = []

    for line in lines:
        if not line:
            continue
        line = line + "\n"
        line_before = ""
        line_after = ""

        # Process lists
        if re.match(r"[\*> ]+ ", line):
            if re.match(r" + ", line):
                if new_lines and new_lines[-1].lstrip().startswith("*"):
                    if linebreak_before_list:
                        line_before = "\n"
            was_list = True
            if re.match(r"\*+ ", line):
                last_bullet = re.match(r"\*+", line).end()
                bullet = line[:last_bullet]
                line = line[last_bullet:]
                line = f"{' '*(len(bullet)-1)*3}*{line}"
            elif re.match(r"\*+>+ ", line):
                last_bullet = re.match(r"\*+", line).end()
                bullet = line[:last_bullet]
                line = line[last_bullet:]
                line = f"{' '*(len(bullet))*3}{line}"
            if was_table:
                if linebreak_after_table:
                    line_before = "\n"
                was_table = False
            was_header = False

        # Process tables
        elif re.match("^\\|.*\\|\n", line):
            if not was_table:
                # Add header separator
                num_columns = len(line.split("|")) - 2
                line_after = "|" + " - |" * num_columns + "\n"
            was_table = True
            if was_list:
                if linebreak_after_list:
                    line_before = "\n"
                was_list = False
            was_header = False

        else:
            # Process headers
            if re.match(r"^\#+ ", line):
                hashtags, line = line.split(" ", 1)
                title = line.strip()
                markdown_subsection = re.sub(r" +", "-", re.sub(r"[^\w ]", "", title.lower()))
                url_subsection = urllib.parse.quote(title.replace(" ", "_"))
                level = len(hashtags)
                if level > 3:
                    line = f"**{title}**\n"
                else:
                    if level > 1:
                        section_url = f"{url}#{url_subsection}"
                    else:
                        if not url:
                            url = f"https://fr.{website_main}.org/wiki/{url_subsection}"
                        section_url = url

                    header_line = f"{hashtags}"
                    if add_urls:
                        header_line += f" [{line.strip()}]({section_url})\n"
                    else:
                        header_line += f" {line.strip()}\n"

                    ilevel = level - 1
                    do_print = False
                    while ilevel >= len(headers):
                        headers.append(None)
                    do_print = headers[ilevel] != header_line
                    headers[ilevel] = header_line

                    if do_print:
                        if markdown_subsection in markdown_subsections:
                            i = 1
                            while markdown_subsection + f"-{i}" in markdown_subsections:
                                i += 1
                            markdown_subsection += f"-{i}"
                        if section_url in section_urls:
                            # No way to link to the second subsection with the same title in Wikipedia (?)
                            header_line = f"{hashtags} {line.strip()}\n"
                        markdown_subsections.append(markdown_subsection)
                        section_urls.append(section_url)
                        headers = headers[: ilevel + 1]
                        line = header_line
                        toc_line = f"[{title}](#{markdown_subsection})"
                        if level > 1:
                            toc_line = f"{' '*(level-2)*3}* " + toc_line
                        toc.append(toc_line)
                    else:
                        continue

                if linebreak_after_header:
                    line_after = "\n"
                if linebreak_before_header and not was_header and not linebreak_between_lines:
                    line_before = "\n"
                was_header = True

            else:
                # Add new lines after simple lines
                if linebreak_between_lines:
                    line_after = "\n"
                was_header = False

            # Add new lines after lists and tables
            if was_list:
                if linebreak_after_list:
                    line_before = "\n"
                was_list = False
            if was_table:
                if linebreak_after_table:
                    line_before = "\n"
                was_table = False

        new_lines.append(line_before + line + line_after)

    toc = "\n".join(toc)
    prefix = f"{toc}\n---\n" if add_toc else ""

    return prefix + "".join(new_lines)


CANDIDATE_LANGUAGES = None


def check_language(
    text,
    candidate_languages=None,
):
    """
    Return probabilities of the language of the text.

    Args:
        text (str): the text to check
        candidate_languages (list): the list of languages to consider (default: all languages supported by langid)
    Returns:
        dict: the dictionary of languages and their scores
    """
    import langid

    # Restrict (or not the list of languages)
    global CANDIDATE_LANGUAGES
    if candidate_languages != CANDIDATE_LANGUAGES:
        langid.set_languages(candidate_languages)
        CANDIDATE_LANGUAGES = candidate_languages

    if text.isupper():
        text = text.lower()
    language_and_scores = langid.rank(text)

    exps = [math.exp(v / max(1, len(text))) for k, v in language_and_scores]
    sum_exps = sum(exps)

    language_and_scores = dict(
        sorted(
            [(k, e / sum_exps) for (k, _), e in zip(language_and_scores, exps)],
            key=lambda item: item[1],
        )
    )

    return language_and_scores


if __name__ == "__main__":
    import argparse
    import multiprocessing
    import os

    import numpy as np
    import pandas as pd
    import tqdm

    from data import (
        DataIteratorConcat,
        DataIteratorParquet,
        DataIteratorWikipedia,
        get_datasets,
    )

    parser = argparse.ArgumentParser(
        description="Clean raw text data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="all",
        help="Which dataset to test",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Folder to dump some example data (before/after)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=min(60, multiprocessing.cpu_count()),
        help="Number of threads to use",
    )
    args = parser.parse_args()
    args.num_samples = 20
    args.ignore_if_exists = True

    def df_parallelize(data, func, threads):
        print(f"Parallelizing {len(data)} items on {threads} cores")
        data_split = np.array_split(data, threads)
        pool = multiprocessing.Pool(threads)
        data = pd.concat(pool.map(func, data_split))
        pool.close()
        pool.join()
        return data

    def get_parquet_datasets(it):
        if isinstance(it, list):
            for data in it:
                for d in get_parquet_datasets(data):
                    yield d
        elif isinstance(it, DataIteratorConcat):
            for data in it.datasets:
                for d in get_parquet_datasets(data):
                    yield d
        elif isinstance(it, (DataIteratorParquet, DataIteratorWikipedia)):
            yield it
        else:
            raise NotImplementedError(f"Cannot get parquet files for {type(it)}")

    datas = list(get_datasets(args.dataset, force_raw=True))

    tqdm.tqdm.pandas()

    for data in get_parquet_datasets(datas):
        parquet_files = data.parquet_files
        postprocess = data.postprocess
        preprocess = data.preprocess
        key = data.key
        name = data.name
        name_slug = re.sub(r"[ :/]", "--", name)
        if postprocess is not None or preprocess is not None:
            for filein in tqdm.tqdm(parquet_files):
                dirname, basename = os.path.split(filein)
                odirname = dirname
                if preprocess is not None:
                    odirname += "_preproc"
                if postprocess is not None:
                    odirname += "_cleaned"
                fileout = os.path.join(odirname, basename)
                if args.ignore_if_exists and os.path.exists(fileout):
                    print(f"Skipping {filein} -> {fileout}")
                    continue
                print(f"Postprocessing {filein} -> {fileout}")
                df = pd.read_parquet(filein)
                if args.folder:
                    examples_before = list(df.iloc[: args.num_samples][key])

                if preprocess:
                    if args.threads == 1:
                        raise NotImplementedError("Preprocessing with threads is not implemented")
                    else:

                        def do_preprocess(d):
                            data = []
                            for _, x in d.iterrows():
                                x = preprocess(x)
                                data.append(x)
                            return pd.concat(data, axis=1).T

                        df = df_parallelize(df, do_preprocess, args.threads)

                if postprocess:
                    if args.threads == 1:
                        df[key] = df[key].progress_map(postprocess)
                    else:

                        def do_postprocess(d):
                            d[key] = d[key].progress_map(postprocess)
                            return d

                        df = df_parallelize(df, do_postprocess, args.threads)

                if args.folder:
                    examples_after = list(df.iloc[: args.num_samples][key])
                    folder = os.path.join(args.folder, "cleaning", os.path.basename(dirname))
                    os.makedirs(folder, exist_ok=True)
                    for i, (before, after) in enumerate(zip(examples_before, examples_after)):
                        print(
                            before,
                            file=open(os.path.join(folder, f"{name_slug}_{i}_A.txt"), "w"),
                        )
                        print(
                            after,
                            file=open(os.path.join(folder, f"{name_slug}_{i}_B.txt"), "w"),
                        )
                os.makedirs(odirname, exist_ok=True)
                try:
                    df.to_parquet(fileout)
                except Exception as err:
                    if os.path.exists(fileout):
                        os.remove(fileout)
                    raise RuntimeError(f"Error writing {fileout}") from err
