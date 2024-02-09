import glob
import hashlib
import json
import logging
import os
import pickle
import random
import warnings

import datasets
import regex as re
from text import (
    clean_discours,
    clean_wikipedia,
    remove_useless_lines,
)

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)

_folder = os.path.dirname(os.path.realpath(__file__))
_the_stack_metadata_file = os.path.join(_folder, "assets", "the-stack-dedup-programming-languages-stats.json")
assert os.path.isfile(_the_stack_metadata_file)

DATA_PATH = os.environ.get("DATA_PATH")
if not DATA_PATH:
    for possible_data_path in [
        "/gpfswork/rech/qgz/commun/data/corpus_openllm",  # Jean-Zay
        "/media/storage0/corpus_openllm",  # koios
        "/data-storage/corpus_openllm",  # biggerboi
    ]:
        if os.path.isdir(possible_data_path):
            DATA_PATH = possible_data_path
            break
    if not DATA_PATH:
        raise RuntimeError("No data path found. You can set it using DATA_PATH environment variable.")


########################################
# Custom Iterators


def tokenizer_dataset(train=True, streaming=True, debug=False, download_only=False, factor=3):
    """
    Iterator that yields texts to train / test a tokenizer
    """

    if download_only:
        streaming = False

    kwargs = dict(
        streaming=streaming,
        subsample_rate=0.5,
        subsample_invert=not train,
    )

    kwargs_wikipedia = kwargs | dict(
        max_docs=10 if debug else 500000 * factor,
    )
    kwargs_code = kwargs | dict(
        max_chars_per_language=10 if debug else 25000000,
    )
    kwargs_gallica = kwargs | dict(
        max_parquet_files=2,
        max_docs=10 if debug else None,
    )

    wikipedia_fr = DataIteratorWikipedia(language="fr", **kwargs_wikipedia)
    if download_only:
        del wikipedia_fr

    wikipedia_en = DataIteratorWikipedia(language="en", **kwargs_wikipedia)
    if download_only:
        del wikipedia_en

    if download_only:
        return

    code = DataIteratorCode(**kwargs_code)

    gallica = DataIteratorGallicaMono(**kwargs_gallica)

    if download_only:
        return
    return (
        f"WikipediaFrEn{factor * 500}kpages-Gallica1G-TheStack{25}Mchars",
        DataIteratorConcat(
            [
                gallica,
                wikipedia_fr,
                wikipedia_en,
                code,
            ]
        ),
    )


def get_datasets(name, **kwargs):
    """
    Iterator that yields one or sevaral datasets

    Args:
        name: the name of the dataset(s)
            examples:
            - "all": all datasets used to train the LLM
            - "tok_train": all datasets used to train the tokenizer
            - "wikipedia": all Wikipedia datasets
            - "wikipedia_fr": French Wikipedia
            - "american_stories": American Stories
        **kwargs: additional arguments to pass to all dataset iterators
    """

    multilang_corpora = ["wikipedia", "gutenberg", "europarl", "claire"]

    name = name.lower()
    if name == "all":
        for name in multilang_corpora + [
            # French
            "gallica_press",
            "gallica_mono",
            "theses",
            "hal",
            "persee",  # Non Commercial
            "open_edition",
            "discours_publics",
            "other_fr",
            # English
            "american_stories",
            # Code
            "code",
        ]:
            for ds in get_datasets(name, **kwargs):
                yield ds

    elif name.startswith("tok"):
        if "test" not in name:
            yield tokenizer_dataset(train=True)
        if "train" not in name:
            yield tokenizer_dataset(train=False)

    elif name in multilang_corpora:
        languages = ["fr", "en"] if (name == "claire") else ["fr", "en", "de", "es"]
        for language in languages:
            for ds in get_datasets(f"{name}_{language}", **kwargs):
                yield ds

    else:
        has_language = any(name.startswith(c + "_") for c in multilang_corpora)
        if has_language:
            name, language = name.split("_", 1)
            kwargs["language"] = language
        camel_name = "".join([w.capitalize() for w in name.split("_")])
        class_name = f"DataIterator{camel_name.capitalize()}"
        python_class = globals().get(f"DataIterator{camel_name}", None)
        if not python_class:
            raise RuntimeError(f"Cannot find python class {class_name}")
        yield (camel_name, python_class(**kwargs))


########################################
# Base dataset iterator classes


class DataIterator:
    def __init__(
        self,
        dataset,
        key="text",
        max_docs=None,
        max_words=None,
        max_chars=None,
        subsample_rate=1,
        subsample_criteria=None,
        subsample_invert=False,
        postprocess=None,
        filter_fn=None,
        name="",
    ):
        """
        Args:
            dataset: a dataset object
            key: the key of the text in the dataset
            max_docs: the maximum number of pages to return
            max_words: the maximum number of words to return
            max_chars: the maximum number of characters to return
            subsample_rate: the rate of subsampling
            subsample_criteria: the key to use for subsampling
            subsample_invert: whether to invert the subsampling
            postprocess: a function to apply to the text
            filter_fn: a function to filter the examples
                    (returns True if the example is to be kept, False otherwise)
            name: the name of the dataset
        """
        self.dataset = dataset
        self.dataset_iter = dataset.__iter__()
        self.key = key
        self.max_docs = max_docs
        self.max_chars = max_chars
        self.max_words = max_words
        self.subsample_rate = subsample_rate
        self.subsample_criteria = subsample_criteria
        self.subsample_invert = subsample_invert
        self.postprocess = postprocess
        self.filter_fn = filter_fn

        self.random_generator = random.Random(42)
        self.idx = 0
        self.name = name
        self.num_chars = 0
        self.num_words = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.idx += 1
        if self.max_docs and self.idx > self.max_docs:
            raise StopIteration
        if self.max_chars and self.num_chars >= self.max_chars:
            raise StopIteration
        if self.max_words and self.num_words >= self.max_words:
            raise StopIteration

        data = next(self.dataset_iter)

        if self.filter_fn:
            while not self.filter_fn(data):
                # Skip this example
                data = next(self.dataset_iter)

        # Subsampling
        if self.subsample_rate and self.subsample_rate < 1:
            if self.subsample_criteria is None:
                r = self.random_generator.random()
            else:
                criterion = data[self.subsample_criteria]
                r = string_to_random01(criterion)
            while (r <= self.subsample_rate) if self.subsample_invert else (r > self.subsample_rate):
                # Skip this example
                data = next(self.dataset_iter)
                if self.filter_fn:
                    while not self.filter_fn(data):
                        # Skip this example
                        data = next(self.dataset_iter)
                if self.subsample_criteria is None:
                    r = self.random_generator.random()
                else:
                    criterion = data[self.subsample_criteria]
                    r = string_to_random01(criterion)

        try:
            text = data[self.key]
        except KeyError as err:
            raise KeyError(f"Key {self.key} not found in {data.keys()}.") from err

        if self.postprocess:
            text = self.postprocess(text)

        if self.max_chars:
            self.num_chars += len(text)

        if self.max_words:
            self.num_words += len(text.split())

        return text

    def __len__(self):
        try:
            len(self.dataset)
        except TypeError:
            return self.max_docs or 0
        if self.subsample_rate and self.subsample_rate < 1:
            # Unknown a priori length?
            if self.max_docs:
                return self.max_docs
            return 0
        if self.max_docs:
            return min(len(self.dataset), self.max_docs)
        return len(self.dataset)


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


class DataIteratorConcat:
    def __init__(self, datasets):
        self.datasets = datasets
        self.idx = 0

    def __iter__(self):
        return self

    def __len__(self):
        if not isinstance(self.datasets, list):
            return 0
        s = 0
        for d in self.datasets:
            if len(d) in [None, 0]:
                return 0
            s += len(d)
        return s

    def __next__(self):
        if self.idx >= len(self.datasets):
            raise StopIteration
        try:
            return next(self.datasets[self.idx])
        except StopIteration:
            self.idx += 1
            return self.__next__()


class DataIteratorParquet(DataIterator):
    def __init__(
        self,
        data_path,
        streaming=True,
        max_parquet_files=None,
        force_raw=False,
        **kwargs,
    ):
        if (
            not force_raw
            and os.path.exists(str(data_path) + "_cleaned")
            and kwargs.get("postprocess", None) is not None
        ):
            data_path = data_path + "_cleaned"
            logger.info(f"Using pre-cleaned in {data_path}")
            kwargs.pop("postprocess", None)
        parquet_files = glob.glob(data_path + "/*parquet")
        if parquet_files and re.match(r".*_\d+\.parquet$", parquet_files[0]):
            parquet_files = sorted(parquet_files, key=lambda x: int(x.split(".")[-2].split("_")[-1]))
        if max_parquet_files and max_parquet_files < len(parquet_files):
            parquet_files = parquet_files[:max_parquet_files]
        logger.info(f"Using {len(parquet_files)} parquet files in {data_path}")
        assert len(parquet_files) > 0, f"No parquet files found in {data_path}"

        self.parquet_files = parquet_files

        DataIterator.__init__(
            self,
            datasets.load_dataset(
                "parquet",
                data_files=parquet_files,
                streaming=streaming,
                split="train",
            ),
            **kwargs,
        )


class DataIteratorParquetSplitted(DataIteratorConcat):
    def __init__(self, data_path, streaming=True, max_parquet_files=None, **kwargs):
        parquet_files = glob.glob(data_path + "/*parquet")
        if parquet_files and re.match(r"_\d+\.parquet$", parquet_files[0]):
            parquet_files = sorted(parquet_files, key=lambda x: int(x.split(".")[-2].split("_")[-1]))
        if max_parquet_files and max_parquet_files < len(parquet_files):
            parquet_files = parquet_files[:max_parquet_files]
        assert len(parquet_files) > 0, f"No parquet files found in {data_path}"

        self.name = kwargs.pop("name", "")

        DataIteratorConcat.__init__(
            self,
            [
                DataIterator(
                    datasets.load_dataset(
                        "parquet",
                        data_files=parquet_file,
                        streaming=streaming,
                        split="train",
                    ),
                    name=f"{self.name}:{os.path.splitext(os.path.basename(parquet_file))[0]}",
                    **kwargs,
                )
                for parquet_file in parquet_files
            ],
        )


########################################
# Datasets: Natural Language


class DataIteratorWikipedia(DataIterator):
    def __init__(self, language="fr", streaming=True, from_huggingface=None, force_raw=False, **kwargs):
        version = "20240201"
        repo = f"OpenLLM-France/wikipedia.{language}"
        name = f"{os.path.basename(repo)}/{version}"

        if force_raw:
            from_huggingface = False

        if from_huggingface is None:
            from_huggingface = not os.path.isdir(f"{DATA_PATH}/wikipedia/{language}")
            logger.info(
                f"Using HuggingFace version for {name}" if from_huggingface else f"Using local version for {name}"
            )

        postprocess = clean_wikipedia

        if from_huggingface:
            kwargs_dataset = dict(name=version)
        else:
            data_path = f"{DATA_PATH}/wikipedia/{language}/{version}"
            if not force_raw and os.path.exists(data_path + "_cleaned"):
                data_path = data_path + "_cleaned"
                logger.info(f"Using pre-cleaned in {data_path}")
                postprocess = None

            repo = "parquet"
            pattern = f"{data_path}/*.parquet"
            self.parquet_files = sorted(glob.glob(pattern))
            assert len(self.parquet_files), f"Missing parquet files for {pattern}"
            logger.info(f"Found {len(self.parquet_files)} parquet files in {os.path.dirname(pattern)}")
            kwargs_dataset = dict(data_files=self.parquet_files)

        DataIterator.__init__(
            self,
            datasets.load_dataset(repo, streaming=streaming, split="train", **kwargs_dataset),
            subsample_criteria="id",
            name=name,
            postprocess=postprocess,
            **kwargs,
        )


class DataIteratorGutenberg(DataIteratorParquet):
    def __init__(
        self,
        language="fr",
        filter_legal=True,
        current_year=2024,
        **kwargs,
    ):
        name = f"Gutenberg.{language}"
        thr = 80 if language == "fr" else 70
        if filter_legal:
            name += f".{thr}"

        def get_age(field, default=None):
            if not field:
                return default
            return int(field)

        def filter_gutenberg(x, collect_stat=False):
            global _stats_gutenberg
            death = get_age(x["authoryearofdeath"])
            birth = get_age(x["authoryearofbirth"])
            author = x["author"]
            if author not in _authors_info:
                _authors_info[author] = (birth, death)
            else:
                if _authors_info[author] != (birth, death):
                    info = f"{_authors_info[author]} != {(birth, death)}"
                    print(f"Author {author} has multiple birth/death dates: {info}")
                    if not death and not birth:
                        birth, death = _authors_info[author]
            if not death and not birth:
                # print(f"Unknown birth/dead date for {author}")
                age_ok = True
                age_ok_for_stats = "?"
            else:
                age_ok = bool(
                    (death and death <= current_year - thr)
                    or (not death and birth and birth <= current_year - thr - 80)
                )
                age_ok_for_stats = age_ok
            copyright_ok = "copyright" not in x["usagerights"]
            if collect_stat:
                key = (language, age_ok_for_stats, x["usagerights"])
                pages, words, chars = _stats_gutenberg.setdefault(key, [0, 0, 0])
                pages += 1
                words += len(x["text"].split())
                chars += len(x["text"])
                _stats_gutenberg[key] = [pages, words, chars]
                key = (language, "TOTAL", "TOTAL")
                pages, words, chars = _stats_gutenberg.setdefault(key, [0, 0, 0])
                pages += 1
                words += len(x["text"].split())
                chars += len(x["text"])
                _stats_gutenberg[key] = [pages, words, chars]
            return age_ok and copyright_ok

        DataIteratorParquet.__init__(
            self,
            f"{DATA_PATH}/gutenberg_parquet/{language}",
            subsample_criteria="id",
            filter_fn=filter_gutenberg if filter_legal else None,
            name=name,
            **kwargs,
        )


_authors_info = {}
_stats_gutenberg = {}


def print_gutenberg_stats():
    if not _stats_gutenberg:
        return
    print("Gutenberg stats:")
    print(
        f"""\
| {'language':8} \
| {'death<70/80':11} \
| {'usage rights':25} \
| {'books':>8} \
| {'words':>14} \
| {'chars':>14} |"""
    )

    def language_order(lan):
        return {"en": 0, "fr": 1, "de": 2, "es": 3}[lan]

    def openness_order(rights):
        return {
            True: 0,
            "?": 0.5,
            False: 1,
            "TOTAL": -1,
            "open": 0,
            "open_restricted": 1,
            "unknown": 2,
            "copyright_open": 3,
            "copyright_open_restricted": 4,
            "copyright": 5,
        }[rights]

    for k in sorted(
        _stats_gutenberg.keys(),
        key=lambda k: [
            language_order(k[0]),
            openness_order(k[1]),
            openness_order(k[2]),
        ],
    ):
        language, age_ok, rights = k
        pages, words, chars = _stats_gutenberg[k]
        age_ok = {True: "OK", False: "no"}.get(age_ok, age_ok)
        print(
            f"""\
| {language:8} \
| {age_ok:11} \
| {rights:25} \
| {formatnum(pages):>8} \
| {formatnum(words):>14} \
| {formatnum(chars):>14} |"""
        )


def formatnum(num):
    # Add non breakable space each 3 digits
    num = str(num)
    return "\u00A0".join(num[::-1][i : i + 3] for i in range(0, len(num), 3))[::-1]


class DataIteratorEuroparl(DataIterator):
    def __init__(self, language="fr", streaming=True, from_roots=False, **kwargs):
        if from_roots:
            repo = f"bigscience-data/roots_{language}_the_pile_europarl"
            kwargs_dataset = {}
        else:
            repo = "text"
            kwargs_dataset = dict(
                data_files=f"{DATA_PATH}/europarl/europarl_{language}_v10.txt",
                sample_by="paragraph",
            )

        DataIterator.__init__(
            self,
            datasets.load_dataset(repo, streaming=streaming, split="train", **kwargs_dataset),
            name=f"Europarl{language.capitalize()}",
            **kwargs,
        )


class DataIteratorClaire(DataIteratorConcat):
    def __init__(self, language="fr", streaming=True, split=None, only_open=True, **kwargs):
        path = DATA_PATH + f"/claire_{language}"
        full_files = glob.glob(path + "/*/full.txt")
        train_files = glob.glob(path + "/*/train.txt")
        test_files = glob.glob(path + "/*/test.txt")
        # Ignore full files if we have train files
        full_files = [f for f in full_files if f.replace("full.txt", "train.txt") not in train_files]
        train_files += full_files

        to_exclude = [
            "AntiScam",
            "CaSiNo",
            "CraigslistBargains",
            "FRAMES",
        ]  # JH: highly redundant, short, fake and just dumb
        train_files = [f for f in train_files if not any(e in f for e in to_exclude)]
        test_files = [f for f in test_files if not any(e in f for e in to_exclude)]

        assert train_files, f"No files found in {path}"
        if split is None:
            files = train_files + test_files
        elif split == "train":
            files = train_files
        elif split == "test":
            files = test_files

        dataset_to_filenames = {}
        for filename in files:
            name = os.path.basename(os.path.dirname(filename))
            if name.startswith("ASR-"):
                name = name[4:]
            name = re.split(r"[\-_]", name)[0]
            if name.startswith("Theatre"):
                name = "Theatre"
            if only_open and language == "fr" and name not in ["Theatre", "AssembleeNationale", "Senat"]:
                continue
            dataset_to_filenames.setdefault(name, []).append(filename)

        self.name = name = f"Claire{language.capitalize()}"
        DataIteratorConcat.__init__(
            self,
            [
                DataIterator(
                    datasets.load_dataset(
                        "text",
                        data_files={"train": filenames},
                        streaming=streaming,
                        sample_by="paragraph",
                        split="train",
                    ),
                    name=f"Claire{language.capitalize()}:{name}" + (f":{split}" if split else ""),
                    **kwargs,
                )
                for name, filenames in dataset_to_filenames.items()
            ],
        )


########################################
# Datasets: French


class DataIteratorGallicaMono(DataIteratorParquet):
    def __init__(self, **kwargs):
        DataIteratorParquet.__init__(
            self,
            DATA_PATH + "/gallica_mono_parquet",
            key="complete_text",
            name="GallicaMonographies",
            # postprocess=clean_pdf_extraction_and_html, # TODO?
            **kwargs,
        )


class DataIteratorGallicaPress(DataIteratorConcat):
    def __init__(self, **kwargs):
        self.name = "GallicaPress"
        DataIteratorConcat.__init__(
            self,
            [
                DataIteratorParquet(
                    DATA_PATH + f"/gallica_presse_{source}_parquet",
                    key="complete_text",
                    name=f"GallicaPress:{source}",
                    # postprocess=
                    #     lambda text: clean_pdf_extraction(text, html_escape=True),
                    **kwargs,
                )
                for source in ("html", "txt")
            ],
        )


class DataIteratorDiscoursPublics(DataIteratorParquet):
    def __init__(self, **kwargs):
        DataIteratorParquet.__init__(
            self,
            DATA_PATH + "/discours_publics_parquet",
            name="DiscoursPublics",
            postprocess=lambda text: clean_discours(text),
            **kwargs,
        )


class DataIteratorHal(DataIteratorParquet):
    def __init__(self, **kwargs):
        DataIteratorParquet.__init__(
            self,
            DATA_PATH + "/hal_parquet",
            name="HAL",
            # postprocess=lambda text: clean_pdf_extraction(text, html_escape=True),
            **kwargs,
        )


class DataIteratorPersee(DataIteratorParquet):
    def __init__(self, **kwargs):
        DataIteratorParquet.__init__(
            self,
            DATA_PATH + "/persee_parquet",
            name="Persee",
            key="complete_text",
            # postprocess=lambda text: clean_pdf_extraction(text, html_escape=True),
            **kwargs,
        )


class DataIteratorTheses(DataIteratorParquet):
    def __init__(self, **kwargs):
        DataIteratorParquet.__init__(
            self,
            DATA_PATH + "/theses_parquet",
            name="Theses",
            # postprocess=lambda text: clean_pdf_extraction(text, html_escape=True),
            key="complete_text",
            **kwargs,
        )


class DataIteratorOpenEdition(DataIteratorParquet):
    def __init__(self, **kwargs):
        DataIteratorParquet.__init__(
            self,
            DATA_PATH + "/open_edition_parquet",
            name="OpenEdition",
            # postprocess=lambda text: clean_pdf_extraction(text, html_escape=True),
            **kwargs,
        )


class DataIteratorOtherFr(DataIteratorParquetSplitted):
    def __init__(self, **kwargs):
        DataIteratorParquetSplitted.__init__(
            self,
            DATA_PATH + "/other_fr_parquet",
            name="OtherFr",
            # postprocess=lambda text: clean_pdf_extraction(text, html_escape=True),
            **kwargs,
        )


########################################
# Datasets: English


class DataIteratorAmericanStories(DataIteratorConcat):
    def __init__(self, streaming=True, from_huggingface=None, **kwargs):
        self.name = "AmericanStories"

        if from_huggingface is None:
            from_huggingface = not os.path.isdir(f"{DATA_PATH}/americanstories")
            logger.info(
                f"Using HuggingFace version for {self.name}"
                if from_huggingface
                else f"Using local version for {self.name}"
            )

        key = "article"

        if not from_huggingface:
            data_files = sorted(glob.glob(f"{DATA_PATH}/americanstories/*.parquet"))
            assert len(data_files), f"Missing parquet files for {DATA_PATH}/americanstories/*.parquet"
            logger.info(f"Using {len(data_files)} parquet files in {DATA_PATH}/americanstories")

            key = "text"

            datas = {
                os.path.splitext(os.path.basename(data_file))[0]: datasets.load_dataset(
                    "parquet",
                    data_files=data_file,
                    streaming=streaming,
                    split="train",
                )
                for data_file in data_files
            }

        else:
            datas = datasets.load_dataset(
                "dell-research-harvard/AmericanStories",
                "all_years",
                streaming=streaming,
            )

        DataIteratorConcat.__init__(
            self,
            [
                DataIterator(
                    datas[year],
                    key=key,
                    subsample_criteria="article_id",
                    name=f"AmericanStories:{year}",
                    postprocess=remove_useless_lines,
                    **kwargs,
                )
                for year in datas.keys()
            ],
        )


class DataIteratorPes2o(DataIteratorConcat):
    def __init__(self, streaming=True, train=None, split_by_type=True, **kwargs):
        repo = "allenai/peS2o"

        if train is not None:
            splits = ["train"] if train else ["validation"]
        else:
            splits = ["validation", "train"]

        if split_by_type:
            filter_fns = {
                "s2ag": lambda x: x["source"].startswith("s2ag"),
                "s2orc": lambda x: not x["source"].startswith("s2ag"),
            }
        else:
            filter_fns = {"": None}

        self.name = "Pes2o"
        DataIteratorConcat.__init__(
            self,
            [
                DataIterator(
                    datasets.load_dataset(
                        repo,
                        streaming=streaming,
                        split=split,
                    ),
                    filter_fn=filter_fn,
                    subsample_criteria="id",
                    name=f"{self.name}:{subset_name+':' if subset_name else ''}{split}",
                    **kwargs,
                )
                for split in splits
                for subset_name, filter_fn in filter_fns.items()
            ],
        )


########################################
# Datasets: Code


_programming_languages = [
    "tex",
    "python",
    "javascript",
    "java",
    "c++",
    "c#",
    "c",
    "php",
    "go",
    "ruby",
    "swift",
    "matlab",
    "julia",
    "typescript",
    "r",
    "rust",
    "scala",
    "dart",
    "haskell",
    "lua",
    "perl",
    "elixir",
    "kotlin",
    "clojure",
    "racket",
    "erlang",
    "ocaml",
    "fortran",
    "assembly",
    "mathematica",
]


class DataIteratorCode(DataIteratorConcat):
    def __init__(
        self,
        streaming=True,
        max_chars_per_language=None,
        from_huggingface=None,
        **kwargs,
    ):
        self.name = "TheStack"

        if from_huggingface is None:
            from_huggingface = not os.path.isdir(f"{DATA_PATH}/the-stack-dedup")
            logger.info(
                f"Using HuggingFace version for {self.name}"
                if from_huggingface
                else f"Using local version for {self.name}"
            )

        metadata = json.load(open(_the_stack_metadata_file, encoding="utf8"))
        # Lower case all keys
        metadata = {k.lower(): v for k, v in metadata.items()}

        # Check we are not missing one language
        for lan in _programming_languages:
            assert lan in metadata.keys(), f"Missing language {lan} in metadata file {list(metadata.keys())}"

        dataset_name = "bigcode/the-stack-dedup" if from_huggingface else "parquet"

        DataIteratorConcat.__init__(
            self,
            list(
                self.yield_datasets(
                    metadata,
                    from_huggingface,
                    dataset_name,
                    streaming,
                    max_chars_per_language,
                    **kwargs,
                )
            ),
        )

    def yield_datasets(
        self,
        **kwargs,
    ):
        for lan in _programming_languages:
            ds = self._get_subset(lan, **kwargs)
            if ds is not None:
                yield ds

    def _get_subset(
        self,
        lan,
        metadata,
        from_huggingface,
        dataset_name,
        streaming,
        max_chars_per_language,
        **kwargs,
    ):
        data_dir = f"data/{metadata[lan]['name']}"

        if from_huggingface:
            kwargs_dataset = dict(data_dir=data_dir)
        else:
            pattern = f"{DATA_PATH}/the-stack-dedup/{data_dir}/*.parquet"
            data_files = sorted(glob.glob(pattern))
            if not len(data_files):
                warnings.warn(f"Missing parquet files for {pattern}", stacklevel=2)
                return None
            logger.info(f"Found {len(data_files)} parquet files in {os.path.dirname(pattern)}")
            kwargs_dataset = dict(data_files=data_files)

        return DataIterator(
            datasets.load_dataset(dataset_name, streaming=streaming, split="train", **kwargs_dataset),
            key="content",
            subsample_criteria="hexsha",
            max_chars=max_chars_per_language,
            filter_fn=(
                (lambda x: x["ext"].lower() == "tex") if lan == "tex" else None
            ),  # Exclude bbl, bib, ... from LaTeX
            name=f"TheStack:{lan}",
            **kwargs,
        )


########################################
# Quick test

if __name__ == "__main__":
    import argparse
    import time

    import tqdm

    random.seed(1234)

    parser = argparse.ArgumentParser(
        description="Test the data iterators and print statistics about datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset",
        nargs="*",
        default=["all"],
        help="Which dataset to test",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Maximum number of samples to iterate on",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Folder to dump some example data into",
    )
    parser.add_argument(
        "--ignore_if_exists",
        action="store_true",
        default=False,
        help="Skip if stat is already computed",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=10,
        help="Number of pages to dump as examples (when --folder is specified)",
    )
    parser.add_argument(
        "--only_dump_examples",
        action="store_true",
        default=False,
        help="To only dump some examples",
    )
    args = parser.parse_args()

    def simple_slugify(name):
        return re.sub(r"[ :/]", "--", name).strip("_-")

    def remove_common_prefix(main, sub):
        common_prefix = os.path.commonprefix([main, sub])
        return sub[len(common_prefix) :]

    def test_iterator(
        it,
        folder=None,
        name="",
        ignore_if_exists=False,
        num_examples=0,
        only_dump_examples=False,
        prefix_example_files=None,
    ):
        name_slug = simple_slugify(name)
        if prefix_example_files is None:
            prefix_example_files = name_slug
        stats = None
        if folder:
            stat_filename = os.path.join(folder, f"stats_{name_slug}.json")
            if os.path.isfile(stat_filename):
                stats = json.load(open(stat_filename, encoding="utf8"))
                if ignore_if_exists and not only_dump_examples:
                    return stats
                num_billion_words = stats["num words"] / 1_000_000_000
                to_insert = f"{num_billion_words:06.3f}B"
                if "--" in prefix_example_files:
                    prefix_example_files = prefix_example_files.replace("--", "--" + to_insert + "_", 1)
                else:
                    prefix_example_files += "--" + to_insert
        tic = time.time()
        i_page = -1
        num_words = 0
        num_chars = 0
        for i_page, text in enumerate(tqdm.tqdm(it, total=len(it))):
            if args.max_examples and i_page >= args.max_examples:
                i_page -= 1
                break
            num_words += len(text.split())
            num_chars += len(text)
            if i_page < num_examples and folder:
                example_folder = os.path.join(folder, "examples")
                os.makedirs(example_folder, exist_ok=True)
                filename = os.path.join(example_folder, f"{prefix_example_files}")
                if num_examples > 1:
                    filename += f"_{i_page:02d}"
                filename += ".txt"
                print(f"Dumping {filename}")
                with open(filename, "w", encoding="utf8") as f:
                    f.write(text)
            elif only_dump_examples:
                break
        if only_dump_examples:
            return {}
        toc = time.time()
        stats = {
            "time to iterate (sec)": toc - tic,
            "num pages": i_page + 1,
            "num words": num_words,
            "num chars": num_chars,
        }
        if folder:
            json.dump(
                stats,
                open(stat_filename, "w", encoding="utf8"),
                indent=2,
                ensure_ascii=False,
            )
        return stats

    def update_stats(global_stats, stats):
        for k, v in stats.items():
            if k not in global_stats:
                global_stats[k] = 0
            global_stats[k] += v

    for dataset in args.dataset:
        for name, it in get_datasets(dataset):
            num_examples = args.num_examples
            if hasattr(it, "name"):
                name = it.name
            name_slug = simple_slugify(name)
            main_prefix_example_files = None
            main_stat_filename = os.path.join(args.folder, f"stats_{name_slug}.json") if args.folder else None
            if main_stat_filename and os.path.isfile(main_stat_filename) and args.only_dump_examples:
                stats = json.load(open(main_stat_filename, encoding="utf8"))
                num_billion_words = stats["num words"] / 1_000_000_000
                main_prefix_example_files = f"{num_billion_words:06.3f}B_{name_slug}"
            # elif args.only_dump_examples:
            #     raise RuntimeError(f"Missing main stat file {main_stat_filename}")

            if isinstance(it, DataIteratorConcat):
                its = it.datasets
                global_stats = {}
            else:
                it.name = name
                its = [it]
                global_stats = None

            max_num_examples_per_subset = num_examples / len(its)
            for subset in its:
                subname = subset.name
                num_examples = int(max_num_examples_per_subset) + (
                    1 if random.random() < (max_num_examples_per_subset % 1) else 0
                )
                if num_examples == 0 and any(s in subname for s in ("tex", "python")):
                    num_examples = 2
                if "other" in name.lower():
                    num_examples = args.num_examples
                if num_examples == 0 and args.only_dump_examples:
                    continue
                print(f"* {subname}")
                if main_prefix_example_files:
                    suffix = remove_common_prefix(name_slug, simple_slugify(subname))
                    prefix_example_files = f"{main_prefix_example_files}{suffix}"
                else:
                    prefix_example_files = None
                stats = test_iterator(
                    subset,
                    folder=args.folder,
                    name=subname,
                    ignore_if_exists=args.ignore_if_exists,
                    num_examples=num_examples,
                    only_dump_examples=args.only_dump_examples,
                    prefix_example_files=prefix_example_files,
                )
                if args.only_dump_examples:
                    continue
                print(json.dumps(stats, indent=4))

                if global_stats is not None:
                    update_stats(global_stats, stats)

            if args.only_dump_examples:
                continue

            if global_stats is not None:
                print(f"* {name}")
                print(json.dumps(global_stats, indent=4))
                if args.folder:
                    json.dump(
                        global_stats,
                        open(main_stat_filename, "w", encoding="utf8"),
                        indent=2,
                        ensure_ascii=False,
                    )

    print_gutenberg_stats()
