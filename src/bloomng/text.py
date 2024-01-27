import html

import regex as re


def clean_pdf_extraction(text, html_escape=False):
    text = remove_page_numbers(text)
    text = remove_useless_lines(text)
    if html_escape:
        text = html_unescape(text)
    text = text.replace(" \n", "\n")
    text = text.strip()
    return text


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


def remove_double_spaces(text):
    return re.sub(r" {2,}", r" ", text)


def clean_discours(text):
    text = re.sub(r"-? ?\d+ VUES?$", "", text).rstrip()
    text = re.sub(r"[Ss]ource [^\n]$", "", text).rstrip()
    text = re.sub(r"([Ss]ource [:a-z0-9_\-, ]+)", " ", text)
    text = re.sub(r"[Ss]ource http[:a-z0-9_\-]+", " ", text)
    text = remove_double_spaces(text)
    return text



def remove_page_numbers(text, verbose=False, pattern=None):
    """
    Try to remove page numbers from a plain text version of a PDF.

    Assumptions:
    - Page numbers increase monotonically
    - Some page numbers may be missing (or undetected with a simple regex on OCR output). Maximum 7 consecutive missing page numbers.
    - Starts at page 2
    - Page numbers occurs at the beginning or at the end of a line (modulo some non-word characters)
    - Limited number of patterns supported : — 2 —, (2), [2], {2}, 2, ... and the same with a total like — 2/10 —, (2/10)

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
        for match in re.finditer(rf"(\n[^\n]*\b{current_page_string}(/\d{{2,5}})?\b[^\n\w]*\n)|(\n[^\n\w]*\b{current_page_string}(/\d{{2,5}})?\b[^\n]*\n)", text):
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
            is_not_too_far = start - min_char_index < max_page_length * (
                current_page_number - last_page_number
            )
            # print(start, match.group().strip(), is_after, is_not_too_far)
            if pattern is not None or (is_after and is_not_too_far):
                if not len(candidates):
                    if use_page_length:
                        # print(f"{current_page_number=} | {avg_length=} | {min_char_index=} | {start-min_char_index=}")
                        min_char_index = min(
                            start, min_char_index + avg_length / 10
                        )
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
            # print(f"Match {match.group()} at {match.start()} -> {match.start() > min_char_index}")
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
        text = (
            text[: start].rstrip() + new_content + text[end :].lstrip()
        )
    return text


_page_patterns = (
    [
        rf"{re.escape(delimiter)}\s*\d{{1,5}}(/\d{{2,5}})?\s*{re.escape(delimiter)}"
        for delimiter in ["_", "-", "—"]
    ]
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


if __name__ == "__main__":

    import argparse
    import os

    import pandas as pd
    import tqdm
    from data import DataIteratorConcat, DataIteratorParquet, get_datasets

    parser = argparse.ArgumentParser(
        description="Clean raw text data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="all",
        # choices=[ "tok_all", "tok_train", "tok_test", "code", "wikipedia", "claire", "gallica_mono", "gallica_press", "discours", "american_stories", ],
        help="Which dataset to test",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Folder to dump some example data (before/after)",
    )
    args = parser.parse_args()
    args.num_samples = 20
    args.ignore_if_exists = True

    def get_parquet_datasets(it):
        if isinstance(it, tuple) and len(it) == 2:
            _, data = it
            for d in get_parquet_datasets(data):
                yield d
        if isinstance(it, list):
            for data in it:
                for d in get_parquet_datasets(data):
                    yield d
        if isinstance(it, DataIteratorConcat):
            for data in it.datasets:
                for d in get_parquet_datasets(data):
                    yield d
        if isinstance(it, DataIteratorParquet):
            yield it

    datas = get_datasets(args.dataset, force_raw=True)

    tqdm.tqdm.pandas()

    for data in get_parquet_datasets(datas):
        parquet_files = data.parquet_files
        postprocess = data.postprocess
        key = data.key
        name = data.name
        name_slug = re.sub(r"[ :/]", "--", name)
        if postprocess is not None:

            for filein in tqdm.tqdm(parquet_files):
                dirname, basename = os.path.split(filein)
                fileout = os.path.join(dirname + "_cleaned", basename)
                if args.ignore_if_exists and os.path.exists(fileout):
                    print(f"Skipping {filein} -> {fileout}")
                    continue
                print(f"Postprocessing {filein} -> {fileout}")
                df = pd.read_parquet(filein)
                if args.folder:
                    examples_before = list(df.iloc[:args.num_samples][key])
                df[key] = df[key].progress_map(postprocess)
                if args.folder:
                    examples_after = list(df.iloc[:args.num_samples][key])
                    folder = os.path.join(args.folder, "cleaning", os.path.basename(dirname))
                    os.makedirs(folder, exist_ok=True)
                    for i, (before, after) in enumerate(zip(examples_before, examples_after)):
                        print(before, file=open(os.path.join(folder, f"{name_slug}_{i}_A.txt"), "w"))
                        print(after, file=open(os.path.join(folder, f"{name_slug}_{i}_B.txt"), "w"))
                os.makedirs(dirname + "_cleaned", exist_ok=True)
                df.to_parquet(fileout)
