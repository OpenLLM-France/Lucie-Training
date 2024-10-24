import argparse

from datatrove.data import Document, DocumentsPipeline
from datatrove.executor import SlurmPipelineExecutor
from datatrove.pipeline.filters import URLFilter
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.formatters import PIIFormatter
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import ParquetWriter
from datatrove.pipeline.writers.disk_base import DiskWriter


def extract_url(data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
    """
    `data` is a generator of Document. You must also return a generator of Document (yield)
    You can optionally use `rank` and `world_size` for sharding
    """
    import json

    for document in data:
        document.metadata["url"] = json.loads(document.metadata["meta"])["url"]
        yield document


class RedPajamaQualityFilter(BaseFilter):
    name = "🔴🦙 RedPajama Quality"

    def __init__(
        self,
        exclusion_writer: DiskWriter = None,
        language: str = "fr",
    ):
        super().__init__(exclusion_writer)
        self.language = language

    def filter(self, doc: Document) -> bool | tuple[bool, str]:  # noqa # C901
        import json

        signals = json.loads(doc.metadata["quality_signals"])

        ### Lucie
        # rule 1: ppl between 10 and 1000
        perplexity = signals["ccnet_perplexity"][0][2]
        if perplexity < 10 or perplexity > 1000:
            return False, "ccnet:perplexity"

        # rule 2: confidence in language > 0.65
        language_score = signals["ccnet_language_score"][0][2]
        if language_score < 0.65:
            return False, "ccnet:language_score"

        ### C4
        # rule: at least 3 sentences
        num_sentences = signals["rps_doc_num_sentences"][0][2]
        if num_sentences < 3:
            return False, "C4:num_sentences"

        # rule: ratio between the number of occurrences of '{' or '}' and the number of characters in the raw text.
        doc_curly_bracket = signals["rps_doc_curly_bracket"][0][2]
        if doc_curly_bracket > 0:
            return False, "C4:curly_bracket"

        # rule: page may not contain placeholder "lorem ipsum" text
        lorem_ipsum = signals["rps_doc_lorem_ipsum"][0][2]
        if lorem_ipsum > 0:
            return False, "C4:lorem_ipsum"

        # TOXICITY
        # rule : page may not contain bad words in bad url
        n_bad_words = signals["rps_doc_ldnoobw_words"][0][2]
        if n_bad_words > 0:
            return False, "C4:toxic_words"

        ### Gopher
        # rule: number of words between 50 and 10'000
        word_count = signals["rps_doc_word_count"][0][2]
        if word_count < 50 or word_count > 100_000:
            return False, "Gopher:word_count"

        # rule: mean word length between 3 and 10
        mean_word_length = signals["rps_doc_mean_word_length"][0][2]
        if mean_word_length < 3 or mean_word_length > 10:
            return False, "Gopher:mean_word_length"

        # rule: symbol to word ratio below 0.1
        symbol_word_ratio = signals["rps_doc_symbol_to_word_ratio"][0][2]
        if symbol_word_ratio > 0.1:
            return False, "Gopher:symbol_word_ratio"

        # rule: 90% of lines need to start without a bullet point
        n_lines = signals["ccnet_nlines"][0][2]
        n_lines_bulletpoint_start = sum(map(lambda ln: ln[2], signals["rps_lines_start_with_bulletpoint"]))
        if n_lines_bulletpoint_start / n_lines > 0.9:
            return False, "Gopher:bulletpoint_start"

        # rule: more than 30% ending with an ellipsis
        lines_end_with_ellipsis_ratio = signals["rps_doc_frac_lines_end_with_ellipsis"][0][2]
        if lines_end_with_ellipsis_ratio > 0.3:
            return False, "Gopher:lines_end_with_ellipsis_ratio"

        # rule: 70% of words in a document contain at least one alphabetic character
        rps_doc_frac_no_alph_words = signals["rps_doc_frac_no_alph_words"][0][2]
        if rps_doc_frac_no_alph_words > 0.3:
            return False, "Gopher_bis:rps_doc_frac_no_alph_words"

        # Gopher repetition removal
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
            return False, "Gopher:rps_doc_frac_chars_top_2gram"
        if rps_doc_frac_chars_top_3gram > 0.18:
            return False, "Gopher:rps_doc_frac_chars_top_3gram"
        if rps_doc_frac_chars_top_4gram > 0.16:
            return False, "Gopher:rps_doc_frac_chars_top_4gram"
        if rps_doc_frac_chars_dupe_5grams > 0.15:
            return False, "Gopher:rps_doc_frac_chars_dupe_5grams"
        if rps_doc_frac_chars_dupe_6grams > 0.14:
            return False, "Gopher:rps_doc_frac_chars_dupe_6grams"
        if rps_doc_frac_chars_dupe_7grams > 0.13:
            return False, "Gopher:rps_doc_frac_chars_dupe_7grams"
        if rps_doc_frac_chars_dupe_8grams > 0.12:
            return False, "Gopher:rps_doc_frac_chars_dupe_8grams"
        if rps_doc_frac_chars_dupe_9grams > 0.11:
            return False, "Gopher:rps_doc_frac_chars_dupe_9grams"
        if rps_doc_frac_chars_dupe_10grams > 0.10:
            return False, "Gopher:rps_doc_frac_chars_dupe_10grams"
        return True


class LucieURLFilter(BaseFilter):
    name = "💻 Lucie URL"

    def __init__(
        self,
        exclusion_writer: DiskWriter = None,
        language: str = "fr",
    ):
        super().__init__(exclusion_writer)
        self.language = language

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        import json

        signals = json.loads(doc.metadata["quality_signals"])

        # rule: url that are already in the other datasets
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

        url = doc.metadata["url"]
        url_dedup = is_url_duplicated(url, self.language)
        if url_dedup:
            return False, "lucie:dedup_url"

        # rule: url blacklist
        rps_doc_ut1_blacklist = signals["rps_doc_ut1_blacklist"][0][2]
        # https://data.together.xyz/redpajama-data-v2/v1.0.0/artifacts/ut1_domain_categories.json
        if rps_doc_ut1_blacklist is not None:
            return False, "lucie:blacklist_url"
        return True


class RedPajamaDuplicatesFilter(BaseFilter):
    name = "👯 RedPajama Duplicates"

    def __init__(
        self,
        exclusion_writer: DiskWriter = None,
    ):
        super().__init__(exclusion_writer)

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        import json

        signals = json.loads(doc.metadata["quality_signals"])

        # Remove duplicates
        if signals["is_duplicate"]:
            return False, "duplicates"
        return True


class CanFetchFilter(BaseFilter):
    name = "👮 Can Fetch URLs"

    def __init__(
        self,
        file_path=None,
        exclusion_writer: DiskWriter = None,
    ):
        super().__init__(exclusion_writer)
        self._valid_domains = None
        self.file_path = file_path

    @property
    def valid_domains(self):
        import json

        if self._valid_domains is None:
            with open(self.file_path) as fp:
                self._valid_domains = set(json.load(fp))
        return self._valid_domains

    def filter(self, doc: Document) -> bool | tuple[bool, str]:  # noqa # C901
        import re
        import urllib.parse

        def canonical_url(url):
            url_base = urllib.parse.urlparse(url).netloc.lower()
            if not url_base and url.strip():
                # extra //
                url = re.sub(r"://+", "://", url)
                url_base = urllib.parse.urlparse(url).netloc.lower()
            return url_base

        url = doc.metadata["url"]
        domain = canonical_url(url)
        if domain in self.valid_domains:
            return True
        else:
            return False, "Cannot fetch this domain"


def get_args():
    parser = argparse.ArgumentParser(description="Process some configurations.")

    # Adding arguments
    parser.add_argument(
        "--dump-to-process", type=str, default="2023-14", help="Specify the dump to process. Default is '2023-14'."
    )
    parser.add_argument("--language", type=str, default="fr", help="Specify the language. Default is 'fr'.")
    parser.add_argument(
        "--main-output-path",
        type=str,
        default="/lustre/fsn1/projects/rech/qgz/uzq54wg/processed_redpajama",
        help="Specify the main output path. Default is '/lustre/fsn1/projects/rech/qgz/uzq54wg/processed_redpajama'.",
    )

    parser.add_argument("--dataset-name", type=str, default="togethercomputer/RedPajama-Data-V2", help="")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    DATASET_NAME = args.dataset_name
    DUMP_TO_PROCESS = args.dump_to_process
    LANGUAGE = args.language
    MAIN_OUTPUT_PATH = args.main_output_path
    FILTERING_OUTPUT_PATH = f"{MAIN_OUTPUT_PATH}/base_processing"

    main_processing_executor = SlurmPipelineExecutor(
        job_name=f"{DUMP_TO_PROCESS}--{LANGUAGE}",
        pipeline=[
            HuggingFaceDatasetReader(
                DATASET_NAME,
                dataset_options={
                    "name": "default",
                    "languages": [LANGUAGE],
                    "snapshots": [DUMP_TO_PROCESS],
                    "partition": "head_middle",
                    "split": "train",
                    "trust_remote_code": True,
                },
                streaming=True,
                text_key="raw_content",
                # limit=1000 # for debug
            ),
            extract_url,
            URLFilter(),
            LucieURLFilter(
                language=LANGUAGE,
            ),
            CanFetchFilter(file_path="/lustre/fsn1/projects/rech/qgz/uzq54wg/valid_domains_redpajama_4500k.json"),
            RedPajamaQualityFilter(
                language=LANGUAGE,
            ),
            RedPajamaDuplicatesFilter(),
            PIIFormatter(email_replacement="<email>", ip_replacement="<ip>"),
            ParquetWriter(f"{FILTERING_OUTPUT_PATH}/output/{LANGUAGE}/{DUMP_TO_PROCESS}"),
        ],
        sbatch_args={"account": "qgz@cpu"},
        tasks=50,  # => 2 minutes
        cpus_per_task=2,
        time="5:00:00",
        logging_dir=f"{MAIN_OUTPUT_PATH}/logs/base_processing/{LANGUAGE}/{DUMP_TO_PROCESS}",
        slurm_logs_folder=f"logs/base_processing/{LANGUAGE}/{DUMP_TO_PROCESS}",  # must be local
        randomize_start_duration=180,  # don't hit the bucket all at once with the list requests
        qos="qos_cpu-t3",
        partition="cpu_p1",
        condaenv="/lustre/fsn1/projects/rech/qgz/uzq54wg/envs/datatrove",
    )

    main_processing_executor.run()
