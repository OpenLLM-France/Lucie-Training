import hashlib
import html
import json
import math
import pickle
import random
import urllib
from collections import Counter
from urllib.parse import urlparse

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
    # Remove lines coming after coma and semi-colon
    text = re.sub(r"([,;])\s*\n", r"\1 ", text)
    # Remove lines when it does not start with a capital letter
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


def clean_eurovoc(text):
    text = re.sub(r" \(cid:1\)", "", text)
    text = re.sub(r"(\w*)\(cid:(\d+)\)(\w*)", _repair_cid_character, text)
    # text = re.sub(r"\f", "", text) # should we remove those control characters?
    return re.sub(r" +", " ", text)


def clean_gutenberg(text):
    def remove_gallica_mention(text):
        return re.sub(".*?http://gallica.bnf.fr.*?\n", "", text)

    def remove_licence(text):
        pattern = r"\n\n            \*\*\* END OF THE PROJECT GUTENBERG EBOOK.*"
        return re.sub(pattern, "", text, flags=re.DOTALL)

    # text = remove_gallica_mention(text)
    text = remove_licence(text)
    return text


### Theses
def clean_theses(text):
    control_char_pattern_except_page_break = r"[\x00-\x08\x0B\x0E-\x1F\x7F�]"

    def remove_HAL(text):
        pattern = r"^.*?HAL is a multi-disciplinary open access.*?\x0c"
        return re.sub(pattern, "", text, flags=re.DOTALL)

    def filter_duplicated_lines(text):
        lines = [text[match.start() : match.end()] for match in re.finditer(r"([^\n\x0c]*[\n\x0c]|[^\n\x0c]+$)", text)]
        count_lines = Counter(lines)
        duplicated_lines = [k for k, v in count_lines.items() if (v >= 10) and (len(k) >= 10)]
        out = []
        for line in lines:
            if line in duplicated_lines:
                pass
            else:
                out.append(line)
        return "".join(out)

    def _remove_pages(text):
        pages = [text[match.start() : match.end()] for match in re.finditer(r"([^\x0c]*[\x0c]|[^\x0c]+$)", text)]
        out = ""
        for page in pages:
            num_lines = len(page.split("\n"))
            frac_control_char = len(re.findall(control_char_pattern_except_page_break, page)) / len(page)
            if num_lines > 200:
                pass
            elif frac_control_char > 0.2:
                pass
            else:
                out += page
        return out

    text = remove_HAL(text)
    text = filter_duplicated_lines(text)
    text = _remove_pages(text)
    text = text.strip()
    return text


def _repair_cid_character(match):  # noqa # C901 `...` is too complex
    i = int(match.group(2))

    if i in _cid_correction_double:
        replacement = _cid_correction_double[i]
        if (
            not match.group(1)
            or not match.group(1)[-1].islower()
            or not match.group(3)
            or not match.group(3)[0].islower()
        ):
            replacement = " "
        j = None
    else:
        j = _cid_correction.get(i, i)
        if j is not None and (i <= 100 or i > 300):
            j = None
        if j is None:
            replacement = " "
        else:
            replacement = chr(j)
    if replacement.islower():
        if not match.group(1):
            if match.group(3) and match.group(3)[0].isupper():
                replacement = " "
            elif replacement not in ["-"] and match.group(3) and match.group(3)[0].isdigit():
                replacement = " "
        elif match.group(1).isupper() and match.group(3).isupper():
            if i in [144]:
                replacement = replacement.upper()
            else:
                replacement = " "
    elif replacement.isupper():
        if not match.group(1):
            if match.group(3) and match.group(3)[0].islower():
                replacement = " "
            elif replacement not in ["-"] and match.group(3) and match.group(3)[0].isdigit():
                replacement = " "
        elif match.group(1).islower() and match.group(3).islower():
            replacement = " "
    converted = match.group(1) + replacement + match.group(3)
    if j and j < 176 and j not in _cid_correction.values():
        if not match.group(1) or not match.group(3):
            return match.group(1) + match.group(3)
    if i in [152]:
        converted = (
            converted.replace("a~", "ã")
            .replace("n~", "ñ")
            .replace("o~", "õ")
            .replace("A~", "Ã")
            .replace("N~", "Ñ")
            .replace("O~", "Õ")
        )
    return converted


_cid_correction_double = {
    5: "ti",
    133: "fi",
    309: "tz",
    415: "ti",
    427: "tt",
}

_cid_correction = {
    106: None,
    107: None,
    108: ord("m"),
    109: None,
    116: ord("q"),
    113: None,
    121: None,
    128: None,
    133: None,
    149: None,
    159: None,
    160: None,
    173: None,
    #
    21: ord("ā"),
    23: ord("z"),
    25: ord("ţ"),
    127: 232,  # è
    129: 252,  # ü
    130: 233,  # é
    131: 226,  # â
    132: 228,  # ä
    134: 229,  # å
    135: 231,  # ç
    136: 234,  # ê
    137: 101,  # e
    138: 232,  # è
    139: 239,  # ï
    140: 238,  # î
    144: 233,  # é
    145: 225,  # á
    146: 237,  # í
    147: 244,  # ô
    148: 246,  # ö
    141: ord("-"),
    150: ord("-"),
    151: ord("-"),
    152: ord("~"),
    155: 243,  # ó
    156: 250,  # ú
    157: 241,  # ñ
    #
    142: 196,  # Ä
    143: 197,  # Å
    153: 214,  # Ö
    154: 220,  # Ü
    158: 209,  # Ñ
    190: ord("ü"),
    217: ord(" "),
    296: ord("ź"),
    298: ord("t"),
    321: ord("o"),
    367: ord("l"),
    371: ord("ł"),
    373: ord("m"),
    374: ord("n"),
    375: ord("n"),
    378: ord("o"),
    381: ord("o"),
    383: ord("o"),
    393: ord("p"),
    395: ord("q"),
    396: ord("r"),
    400: ord("s"),
    404: ord("s"),
    407: ord("a"),
    410: ord("t"),
    417: ord("i"),  # u?
    425: ord("t"),
    431: ord("i"),
    437: ord("u"),
    448: ord("v"),
}


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


def load_bad_words(language):
    target_url = f"https://raw.githubusercontent.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/master/{language}"
    response = urllib.request.urlopen(target_url)
    bad_words = response.read().decode("utf-8")
    bad_words = [bad_word.lower() for bad_word in bad_words.split("\n") if bad_word != "" and " " not in bad_word]
    return set(bad_words)


_bad_words = {}


def is_obscene(text, language):
    global _bad_words
    if language not in _bad_words:
        _bad_words[language] = load_bad_words(language)
    bad_words = _bad_words[language]

    words = set(re.sub(r"[^\p{L} ]+", "", text).split())
    number_of_obscene_words = len(words.intersection(bad_words))
    return number_of_obscene_words >= 3


def canonical_url(url):
    return urlparse(url).netloc


def is_url_duplicated(url, language):
    if language == "fr":
        keywords = ["fr.wikipedia", "wiktionary", "wikisource", "theses.fr"]
    elif language == "en":
        keywords = [
            "en.wikipedia",
            "arxiv.org",
            "www.ncbi.nlm.nih.gov/pmc",
            "philpapers.org",
            "exporter.nih.gov",
            "irclogs.ubuntu.com",
            "courtlistener.com",
            "uspto.gov",
        ]
    else:
        keywords = ["wikipedia", "europarl", "op.europa.eu"]
    return any(keyword in url for keyword in keywords)


def lucie_rules_pass_for_redpajama(sample, language) -> bool:  # noqa # C901 `...` is too complex
    """function returns True if the sample complies with Gopher rules"""
    signals = json.loads(sample["quality_signals"])

    ### Lucie
    # rule 1: ppl between 10 and 1000
    perplexity = signals["ccnet_perplexity"][0][2]
    if perplexity < 10 or perplexity > 1000:
        return False, "lucie: perplexity"

    # rule 2: confidence in language > 0.65
    language_score = signals["ccnet_language_score"][0][2]
    if language_score < 0.7:
        return False, "lucie: language score"

    # rule 3: similarity with wikipedia
    wikipedia_score = signals["rps_doc_ml_wikipedia_score"][0][2]
    if wikipedia_score < 0.2:
        return False, "lucie: wikipedia score"

    rps_doc_ut1_blacklist = signals["rps_doc_ut1_blacklist"][0][2]
    # https://data.together.xyz/redpajama-data-v2/v1.0.0/artifacts/ut1_domain_categories.json
    if rps_doc_ut1_blacklist is not None:
        return False, "lucie: blacklist"

    url = json.loads(sample["meta"])["url"]
    url_dedup = is_url_duplicated(url, language)
    if url_dedup:
        return False, "lucie: deduplicated url"

    ### C4
    # rule 1: at least 3 sentences
    num_sentences = signals["rps_doc_num_sentences"][0][2]
    if num_sentences < 3:
        return False, "C4: num sentences"

    # rule 2: page may not contain bad words in bad url
    n_bad_words = signals["rps_doc_ldnoobw_words"][0][2]
    if n_bad_words > 3:  # C4 is 0
        return False, "modified C4: toxic words"

    # rule 3: page may not contain placeholder "lorem ipsum" text
    lorem_ipsum = signals["rps_doc_lorem_ipsum"][0][2]
    if lorem_ipsum > 0:
        return False, "C4: lorem ipsum"

    ### Gopher
    # rule 1: number of words between 50 and 10'000
    word_count = signals["rps_doc_word_count"][0][2]
    if word_count < 50 or word_count > 100_000:
        return False, "Gopher: word count"

    # rule 2: mean word length between 3 and 10
    mean_word_length = signals["rps_doc_mean_word_length"][0][2]
    if mean_word_length < 3 or mean_word_length > 10:
        return False, "Gopher: mean word length"

    # rule 2: symbol to word ratio below 0.1
    symbol_word_ratio = signals["rps_doc_symbol_to_word_ratio"][0][2]
    if symbol_word_ratio > 0.1:
        return False, "Gopher: symbol word ratio"

    # rule 3: 90% of lines need to start without a bullet point
    n_lines = signals["ccnet_nlines"][0][2]
    n_lines_bulletpoint_start = sum(map(lambda ln: ln[2], signals["rps_lines_start_with_bulletpoint"]))  # noqa # C901 `...` is too complex
    if n_lines_bulletpoint_start / n_lines > 0.9:
        return False, "Gopher: bulletpoint start"

    # rule 4: more than 30% ending with an ellipsis
    lines_end_with_ellipsis_ratio = signals["rps_doc_frac_lines_end_with_ellipsis"][0][2]
    if lines_end_with_ellipsis_ratio > 0.3:
        return False, "Gopher: lines_end_with_ellipsis_ratio"

    # rule 5: 70% of words in a document contain at least one alphabetic character
    rps_doc_frac_no_alph_words = signals["rps_doc_frac_no_alph_words"][0][2]
    if rps_doc_frac_no_alph_words > 0.3:  # gopher is 0.2
        return False, "modified Gopher: rps_doc_frac_no_alph_words"

    # Repetition removal
    rps_doc_frac_chars_top_2gram = signals["rps_doc_frac_chars_top_2gram"][0][2]
    rps_doc_frac_chars_top_3gram = signals["rps_doc_frac_chars_top_3gram"][0][2]
    rps_doc_frac_chars_top_4gram = signals["rps_doc_frac_chars_top_4gram"][0][2]
    rps_doc_frac_chars_dupe_5grams = signals["rps_doc_frac_chars_dupe_5grams"][0][2]
    rps_doc_frac_chars_dupe_6grams = signals["rps_doc_frac_chars_dupe_6grams"][0][2]
    rps_doc_frac_chars_dupe_7grams = signals["rps_doc_frac_chars_dupe_7grams"][0][2]
    rps_doc_frac_chars_dupe_8grams = signals["rps_doc_frac_chars_dupe_8grams"][0][2]
    rps_doc_frac_chars_dupe_9grams = signals["rps_doc_frac_chars_dupe_9grams"][0][2]
    rps_doc_frac_chars_dupe_10grams = signals["rps_doc_frac_chars_dupe_10grams"][0][2]
    if rps_doc_frac_chars_top_2gram > 0.2:
        return False, "Gopher: rps_doc_frac_chars_top_2gram"
    if rps_doc_frac_chars_top_3gram > 0.18:
        return False, "Gopher: rps_doc_frac_chars_top_3gram"
    if rps_doc_frac_chars_top_4gram > 0.16:
        return False, "Gopher: rps_doc_frac_chars_top_4gram"
    if rps_doc_frac_chars_dupe_5grams > 0.15:
        return False, "Gopher: rps_doc_frac_chars_dupe_5grams"
    if rps_doc_frac_chars_dupe_6grams > 0.14:
        return False, "Gopher: rps_doc_frac_chars_dupe_6grams"
    if rps_doc_frac_chars_dupe_7grams > 0.13:
        return False, "Gopher: rps_doc_frac_chars_dupe_7grams"
    if rps_doc_frac_chars_dupe_8grams > 0.12:
        return False, "Gopher: rps_doc_frac_chars_dupe_8grams"
    if rps_doc_frac_chars_dupe_9grams > 0.11:
        return False, "Gopher: rps_doc_frac_chars_dupe_9grams"
    if rps_doc_frac_chars_dupe_10grams > 0.10:
        return False, "Gopher: rps_doc_frac_chars_dupe_10grams"

    # Remove duplicates
    if signals["is_duplicate"]:
        return False, "lucie: is_duplicate"
    return True, ""


def string_to_random01(x):
    # Get the hash value of the input string
    # hash_value = hash(str(x))
    hash_value = int(hashlib.sha256(pickle.dumps(str(x))).hexdigest(), 16)

    # Normalize the hash value to the range [0, 1]
    # normalized_value = (hash_value & 0xFFFFFFFF) / 0xFFFFFFFF
    normalized_value = (
        hash_value & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
    ) / 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF

    return normalized_value


def string_to_random_range(x, mini, maxi):
    return round(mini - 0.5 + string_to_random01(x) * (maxi + 1 - mini))


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
