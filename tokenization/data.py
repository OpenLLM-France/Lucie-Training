import functools
import glob
import hashlib
import json
import logging
import os
import pickle
import random
import types
import warnings

import datasets
import regex as re
from text import (
    check_language,
    clean_discours,
    clean_eurovoc,
    clean_wikipedia,
    fix_legi,
    fix_legi_and_remove_title,
    html_unescape,
)

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)

_folder = os.path.dirname(os.path.realpath(__file__))
_the_stack_metadata_file = os.path.join(
    os.path.dirname(_folder), "assets", "the-stack-dedup-programming-languages-stats.json"
)
assert os.path.isfile(_the_stack_metadata_file)

DATA_PATH = os.environ.get("DATA_PATH")
if not DATA_PATH:
    for possible_data_path in [
        "/gpfswork/rech/qgz/commun/data/corpus_openllm",  # Jean-Zay
        "/media/storage0/corpus_openllm",  # koios
        "/data-storage/storage0/corpus_openllm",  # biggerboi
    ]:
        if os.path.isdir(possible_data_path):
            DATA_PATH = possible_data_path
            break
    if not DATA_PATH:
        raise RuntimeError("No data path found. You can set it using DATA_PATH environment variable.")


########################################
# Custom Iterators


def tokenizer_dataset(
    train=True,
    less_minority_languages=True,
    streaming=True,
    debug=False,
    factor=3,
    more_data=False,
    use_bilingual=False,
):
    """
    Iterator that yields texts to train / test a tokenizer
    """

    kwargs = dict(
        streaming=streaming,
        subsample_rate=0.5,
        subsample_invert=not train,
        max_docs=10 if debug else None,
    )

    programming_languages = ["tex", "python", "c++", "javascript"]
    if train:
        programming_languages += ["java", "c", "php", "c#", "go", "typescript", "perl"]
    max_char_base = int(2e9) * factor
    wikipedia_char_base = max_char_base
    code_char_base = int(2e8)
    if more_data:
        wikipedia_char_base = int(1e20)

    kwargs_wikipedia = kwargs | (
        dict(
            max_chars=10 if debug else wikipedia_char_base,
        )
        if train
        else dict(
            max_docs=10 if debug else 500000 * factor,
        )
    )
    kwargs_wikipedia_minority = kwargs_wikipedia.copy()
    if train and less_minority_languages:
        kwargs_wikipedia_minority["max_chars"] = 10 if debug else 200000000 * factor
        # if more_data:
        #     kwargs_wikipedia_minority["max_chars"] *= 5
    kwargs_code = kwargs | (
        dict(
            max_chars_per_language=10 if debug else code_char_base,
            programming_languages=programming_languages,
        )
        if train
        else dict(
            max_chars_per_language=10 if debug else 25000000,
            programming_languages=programming_languages,
        )
    )
    kwargs_gallica = kwargs | dict(
        max_parquet_files=2,
    )
    kwargs_persee = kwargs | dict(
        max_parquet_files=1,
        offset_parquet_files=0 if train else 1,
    )
    kwargs_gutenberg = kwargs | dict(
        max_parquet_files=4,
    )

    all_data = []
    nickname = ""

    if less_minority_languages:
        all_data += list(get_datasets("wikipedia_fr", **kwargs_wikipedia))
        all_data += list(get_datasets("wikipedia_de", **kwargs_wikipedia_minority))
        all_data += list(get_datasets("wikipedia_en", **kwargs_wikipedia))
        all_data += list(get_datasets("wikipedia_es", **kwargs_wikipedia_minority))
        all_data += list(get_datasets("wikipedia_it", **kwargs_wikipedia_minority))
        nickname += f"Wikipedia{factor * 2}BcharsLessMinorityWithIt"
    else:
        all_data += list(get_datasets("wikipedia", **kwargs_wikipedia))
        nickname += f"Wikipedia{factor * 2}Bchars"

    if not train:
        all_data += list(get_datasets("gutenberg", **kwargs_gutenberg))
        nickname += "-Gutenberg"
        nickname = "TEST-" + nickname

    all_data += list(get_datasets("persee", **kwargs_persee))
    nickname += "-Persee"
    all_data += list(get_datasets("legi", postprocess=fix_legi_and_remove_title, **kwargs))
    nickname += "-LEGI"
    all_data += list(get_datasets("europarl", **kwargs))
    nickname += "-Europarl"

    all_data += list(get_datasets("gallica_mono", **kwargs_gallica))
    nickname += "-GallicaMono"

    all_data += list(get_datasets("code", **kwargs_code))
    nickname += "-CodePlus200m"

    if more_data and train:
        all_data += list(get_datasets("theses", max_chars=max_char_base, **kwargs))
        all_data += list(get_datasets("open_data_fr", max_chars=max_char_base, **kwargs))
        all_data += list(get_datasets("claire_fr", subset_regex="theatre", max_chars=max_char_base, **kwargs))
        all_data += list(get_datasets("claire_en", subset_regex="mediasum", max_chars=max_char_base, **kwargs))
        all_data += list(get_datasets("american_stories", max_chars=int(max_char_base / 167), **kwargs))
        all_data += list(get_datasets("pes2o", max_chars=max_char_base, **kwargs))
        all_data += list(get_datasets("eurovoc", max_chars=int(max_char_base / 5), **kwargs))
        nickname = "DatasetV2"
    else:
        nickname = "DatasetV1"

    if use_bilingual and train:
        all_data = list(get_datasets("croissant_aligned", train=True, augment_train=False, **kwargs)) + all_data
        nickname += "-Aligned"

    dataset = DataIteratorConcat(all_data)
    print(f"{'Train' if train else 'Evaluate'} on: {dataset.name}")
    dataset.name = nickname

    return dataset


def get_datasets(name, use_nc=True, **kwargs):  # noqa # C901 `...` is too complex
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
        use_nc: Use non-commercial datasets
        **kwargs: additional arguments to pass to all dataset iterators
    """

    if isinstance(name, list):
        for n in name:
            for ds in get_datasets(n, **kwargs):
                yield ds
        return

    multilang_corpora = ["wikipedia", "wikiother", "gutenberg", "europarl", "claire", "eurovoc"]

    name = name.lower()

    if name.startswith("claire"):
        kwargs["use_nc"] = use_nc

    if name == "all":
        for name in (
            multilang_corpora
            + [
                # French
                "gallica_press",
                "gallica_mono",
                "theses",
                "hal",
                "open_edition",
                "discours_publics",
                "other_fr",
                "open_data_fr",
            ]
            + (["persee"] if use_nc else [])
            + [
                # English
                "american_stories",
                "pes2o",
                # Multi-language (with switching)
                "europarl_aligned",
                "croissant_aligned",
                # Code
                "code",
            ]
        ):
            for ds in get_datasets(name, **kwargs):
                yield ds

    elif name.startswith("tok"):
        if "test" not in name:
            yield tokenizer_dataset(train=True)
        if "train" not in name:
            yield tokenizer_dataset(train=False)

    elif name == "legi":
        DataIteratorOtherFr(regex_parquet="legi", **kwargs)

    elif name in multilang_corpora:
        languages = {
            "claire": ["fr", "en"],
            "wikiother": ["fr"],
            # "wikipedia": ["fr", "en", "de", "es", "it"],
            # "gutenberg": ["fr", "en", "de", "es", "it"],
            "europarl": ["fr", "en", "de", "es"],
            "eurovoc": ["en", "de", "es", "it"],
        }.get(name, ["fr", "en", "de", "es", "it"])
        for language in languages:
            for ds in get_datasets(f"{name}_{language}", **kwargs):
                yield ds
            # yield DataIteratorConcat(
            #     list(get_datasets(f"{name}_{language}", **kwargs)),
            #     name=name.capitalize(),
            # )

    else:
        has_language = any(name.startswith(c + "_") for c in multilang_corpora) and not name.endswith("_aligned")
        if has_language:
            name, language = name.split("_", 1)
            kwargs["language"] = language
        camel_name = "".join([w.capitalize() for w in name.split("_")])
        class_name = f"DataIterator{camel_name}"
        python_class = globals().get(f"DataIterator{camel_name}", None)
        if not python_class:
            raise RuntimeError(f"Cannot find python class {class_name}")
        yield python_class(**kwargs)


def decompose_datasets(dataset, parquet_level=False, return_json_file_if_possible=False):  # noqa # C901 `...` is too complex
    # For recursive calls
    kwargs = dict(
        parquet_level=parquet_level,
        return_json_file_if_possible=return_json_file_if_possible,
    )
    if return_json_file_if_possible and hasattr(dataset, "json_files"):
        for i, jsonl_file in enumerate(sorted(dataset.json_files)):
            yield (f"{dataset.name}--{i:04d}", jsonl_file)

    elif isinstance(dataset, (list, types.GeneratorType)):
        for d in dataset:
            for ds in decompose_datasets(d, **kwargs):
                yield ds

    elif isinstance(dataset, DataIteratorConcat):
        for d in dataset.datasets:
            for ds in decompose_datasets(d, **kwargs):
                yield ds

    elif isinstance(dataset, DataIterator):
        if parquet_level and hasattr(dataset, "parquet_files"):
            for i, parquet_file in enumerate(sorted(dataset.parquet_files)):
                assert not dataset.max_docs
                assert not dataset.max_words
                assert not dataset.max_chars
                assert dataset.subsample_rate == 1
                assert not dataset.subsample_invert
                yield DataIterator(
                    datasets.load_dataset(
                        "parquet",
                        data_files=[parquet_file],
                        streaming=True,
                        split="train",
                    ),
                    name=f"{dataset.name}:{i:03d}",
                    key=dataset.key,
                    preprocess=dataset.preprocess,
                    postprocess=dataset.postprocess,
                    filter_fn=dataset.filter_fn,
                )
        else:
            yield dataset
    else:
        raise ValueError(f"Unknown type {type(dataset)}")


########################################
# Base dataset iterator classes


class DataIteratorBase:
    def __init__(self, name):
        assert name, "Name cannot be empty"
        self.name = name

    def __iter__(self):
        return self

    def __len__(self):
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError


class DataIterator(DataIteratorBase):
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
        preprocess=None,
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
            preprocess: a function to apply to the data
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
        if not self.subsample_criteria:
            # Fallback to text
            self.subsample_criteria = key
        self.subsample_invert = subsample_invert
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.filter_fn = filter_fn

        self.random_generator = random.Random(42)
        self.idx = 0
        self.num_chars = 0
        self.num_words = 0

        suffix = ""
        if max_docs:
            suffix += f"-{max_docs}docs"
        if max_words:
            suffix += f"-{max_words}words"
        if max_chars:
            suffix += f"-{max_chars}chars"
        if subsample_rate and subsample_rate < 1:
            suffix += f"-ssample{subsample_rate}"
            if subsample_invert:
                suffix += "inv"
        if suffix:
            name += ":" + suffix.strip("-")

        DataIteratorBase.__init__(self, name)

    def __iter__(self):
        return self

    def _get_next(self):
        try:
            return next(self.dataset_iter)
        except TypeError as err:
            # Sometimes this can occur because of empty transcription:
            # TypeError: Couldn't cast array of type binary to null
            warnings.warn(f"Got an exception {err}", stacklevel=2)
            return self._get_next()

    def __next__(self):  # noqa # C901 `...` is too complex
        self.idx += 1
        if self.max_docs and self.idx > self.max_docs:
            raise StopIteration
        if self.max_chars and self.num_chars >= self.max_chars:
            raise StopIteration
        if self.max_words and self.num_words >= self.max_words:
            raise StopIteration

        data = self._get_next()

        if self.filter_fn:
            while not self.filter_fn(data):
                # Skip this example
                data = self._get_next()

        # Subsampling
        if self.subsample_rate and self.subsample_rate < 1:
            if self.subsample_criteria is None:
                r = self.random_generator.random()
            else:
                criterion = data[self.subsample_criteria]
                r = string_to_random01(criterion)
            while (r <= self.subsample_rate) if self.subsample_invert else (r > self.subsample_rate):
                # Skip this example
                data = self._get_next()
                if self.filter_fn:
                    while not self.filter_fn(data):
                        # Skip this example
                        data = self._get_next()
                if self.subsample_criteria is None:
                    r = self.random_generator.random()
                else:
                    criterion = data[self.subsample_criteria]
                    r = string_to_random01(criterion)

        if self.preprocess:
            data = self.preprocess(data)
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

        if not text:
            # Empty text
            self.idx -= 1
            return self.__next__()

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


class DataIteratorConcat(DataIteratorBase):
    def __init__(self, datasets, name=None):
        self.datasets = datasets
        self.idx = 0
        if name is None:
            name = "+".join(d.name for d in datasets)
        DataIteratorBase.__init__(self, name)

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
        offset_parquet_files=0,
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
        if os.path.isfile(data_path):
            parquet_files = data_path
            self.parquet_files = [parquet_files]
        else:
            parquet_files = glob.glob(data_path + "/*parquet")
            if parquet_files and re.match(r".*_\d+\.parquet$", parquet_files[0]):
                parquet_files = sorted(parquet_files, key=lambda x: int(x.split(".")[-2].split("_")[-1]))
            elif parquet_files and re.match(r".*\-\d+\-of\-\d+\.parquet$", parquet_files[0]):
                parquet_files = sorted(parquet_files, key=lambda x: int(x.split("-")[-3]))
            else:
                parquet_files = sorted(parquet_files)
            if max_parquet_files and max_parquet_files < len(parquet_files):
                parquet_files = parquet_files[offset_parquet_files : max_parquet_files + offset_parquet_files]
            elif offset_parquet_files:
                parquet_files = parquet_files[offset_parquet_files:]
            logger.info(f"Using {len(parquet_files)} parquet files in {data_path}")
            assert len(parquet_files) > 0, f"No parquet files found in {data_path}"
            self.parquet_files = parquet_files

        name = kwargs.pop("name", "")
        if offset_parquet_files or max_parquet_files:
            if not max_parquet_files:
                max_parquet_files = len(parquet_files) - offset_parquet_files
            name += f":{offset_parquet_files}-{max_parquet_files+offset_parquet_files}parquet"

        DataIterator.__init__(
            self,
            datasets.load_dataset(
                "parquet",
                data_files=parquet_files,
                streaming=streaming,
                split="train",
            ),
            name=name,
            **kwargs,
        )


class DataIteratorParquetSplitted(DataIteratorConcat):
    def __init__(self, data_path, regex_parquet=None, streaming=True, max_parquet_files=None, **kwargs):
        parquet_files = glob.glob(data_path + "/*parquet")
        if parquet_files and re.match(r"_\d+\.parquet$", parquet_files[0]):
            parquet_files = sorted(parquet_files, key=lambda x: int(x.split(".")[-2].split("_")[-1]))
        if max_parquet_files and max_parquet_files < len(parquet_files):
            parquet_files = parquet_files[:max_parquet_files]
        if regex_parquet:
            parquet_files = [f for f in parquet_files if re.search(regex_parquet, f, re.IGNORECASE)]
        assert len(parquet_files) > 0, f"No parquet files found in {data_path}"
        logger.info(f"Using {len(parquet_files)} parquet files in {data_path}")
        assert len(parquet_files) > 0, f"No parquet files found in {data_path}"
        self.parquet_files = parquet_files

        name = kwargs.pop("name", "")
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
                    name=f"{name}:{os.path.splitext(os.path.basename(parquet_file))[0]}",
                    **kwargs,
                )
                for parquet_file in parquet_files
            ],
            name=name,
        )


########################################
# Datasets: Natural Language


class DataIteratorWikipedia(DataIterator):
    def __init__(self, language="fr", streaming=True, from_huggingface=None, force_raw=False, **kwargs):
        version = "20240201"
        repo = f"OpenLLM-France/wikipedia.{language}"
        name = f"Wikipedia:{language}"

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


class DataIteratorWikiother(DataIteratorConcat):
    def __init__(self, language="fr", streaming=True, **kwargs):
        name = f"Wikiother:{language}"

        folder = f"{DATA_PATH}/wikiother/{language}"
        if not os.path.isdir(folder):
            raise RuntimeError(f"Folder {folder} does not exist")

        parquet_per_subfolder = {}

        for subfolder in os.listdir(folder):
            if not os.path.isdir(f"{folder}/{subfolder}"):
                continue
            for root, _, files in os.walk(f"{folder}/{subfolder}"):
                for file in files:
                    if file.endswith(".parquet"):
                        parquet_per_subfolder.setdefault(subfolder, []).append(f"{root}/{file}")

        postprocess = clean_wikipedia

        DataIteratorConcat.__init__(
            self,
            [
                DataIterator(
                    datasets.load_dataset(
                        "parquet",
                        data_files={"train": filenames},
                        streaming=streaming,
                        split="train",
                    ),
                    name=f"{name}:{subname}",
                    postprocess=postprocess,
                    subsample_criteria="id",
                    **kwargs,
                )
                for subname, filenames in parquet_per_subfolder.items()
            ],
            name=name,
        )


class DataIteratorGutenberg(DataIteratorParquet):
    def __init__(  # noqa # C901 `...` is too complex
        self,
        language="fr",
        filter_legal=True,
        current_year=2024,
        **kwargs,
    ):
        name = f"Gutenberg:{language}"
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
            copyright_ok = "copyright" != x["usagerights"]
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
            if language == "en":
                return copyright_ok
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

        name = f"Europarl:{language}"
        DataIterator.__init__(
            self,
            datasets.load_dataset(repo, streaming=streaming, split="train", **kwargs_dataset),
            name=name,
            **kwargs,
        )


class DataIteratorEuroparlAligned(DataIteratorConcat):
    def __init__(self, **kwargs):
        parquets = glob.glob(f"{DATA_PATH}/europarl/*parquet")
        assert len(parquets), f"No parquet files found in {DATA_PATH}/europarl"

        parquets = {os.path.basename(p).split(".")[-2]: p for p in parquets}

        DataIteratorConcat.__init__(
            self,
            [
                DataIteratorParquet(
                    parquet_file,
                    name=f"EuroparlAligned:{languages}",
                    preprocess=create_augmented_text_from_aligned_data,
                    subsample_criteria="id",
                    **kwargs,
                )
                for languages, parquet_file in parquets.items()
            ],
            name="EuroparlAligned",
        )


class DataIteratorCroissantAligned(DataIteratorConcat):
    def __init__(self, train=True, augment_train=True, **kwargs):
        if train is not None:
            splits = ["train"] if train else ["test"]
        else:
            splits = ["train", "test"]

        name = "CroissantAligned"
        do_augment = train
        if do_augment:
            name += "-augmented"

        precomputed_landid = True

        def is_augmented(split):
            return augment_train and split == "train"

        def get_data_path(split):
            return os.path.join(
                DATA_PATH,
                "croissant_aligned",
                f"{split}{'_preproc' if (precomputed_landid and is_augmented(split)) else ''}",
            )

        DataIteratorConcat.__init__(
            self,
            [
                DataIteratorParquet(
                    get_data_path(split),
                    name=f"CroissantAligned:{split}" + ("-augmented" if is_augmented(split) else ""),
                    preprocess=(
                        create_augmented_text_from_aligned_data
                        if precomputed_landid
                        else functools.partial(analyze_bilingual_french_english_data, add_language_in_data=True)
                    )
                    if is_augmented(split)
                    else None,
                    # postprocess=analyze_bilingual_french_english_data if is_augmented(split) else None,
                    subsample_criteria="id",
                    **kwargs,
                )
                for split in splits
            ],
            name="CroissantAligned",
        )


def analyze_bilingual_french_english_data(data, add_language_in_data=False):  # noqa # C901 `...` is too complex
    if add_language_in_data:
        text = data["text"]
    else:
        text = data

    fields = re.split(r"( {2,}|\n{2,}|\t)", text.strip())
    assert (len(fields) + 1) % 2 == 0, f"Got even number of fields {len(fields)} in {fields}"
    if len(fields) > 3:
        text_fields = fields[::2]
        proba_lans = [check_language(text, ["fr", "en"]) for text in text_fields]
        score_regroupements = [
            max(
                sum(p["en"] for p in proba_lans[:i_cut]) + sum(p["fr"] for p in proba_lans[i_cut:]),
                sum(p["fr"] for p in proba_lans[:i_cut]) + sum(p["en"] for p in proba_lans[i_cut:]),
            )
            + 1
            - abs(len(text_fields[:i_cut]) - len(text_fields[i_cut:]))
            / max(len(text_fields[:i_cut]), len(text_fields[i_cut:]))
            for i_cut in range(1, len(text_fields))
        ]
        argmax = score_regroupements.index(max(score_regroupements)) + 1
        # sep = fields[2 * argmax - 1]
        # if len(sep) == 2: sep = sep[0]
        # else: sep = " "
        new_fields = ["".join(fields[: 2 * argmax - 1]), fields[2 * argmax - 1], "".join(fields[2 * argmax :])]
        print(f"WARNING: regrouped\n{fields}\n---\ninto\n---\n{new_fields}")
        fields = new_fields

    assert len(fields) == 3, f"Got {len(fields)} in {fields}"
    text1, separator, text2 = fields

    score1 = check_language(text1, ["fr", "en"])
    score2 = check_language(text2, ["fr", "en"])
    gap1 = score1["en"] - score1["fr"]
    gap2 = score2["en"] - score2["fr"]
    if gap1 > gap2:
        lan1, lan2 = "en", "fr"
    else:
        lan1, lan2 = "fr", "en"

    if add_language_in_data:
        data["text_en"] = text1.strip() if lan1 == "en" else text2.strip()
        data["text_fr"] = text1.strip() if lan1 == "fr" else text2.strip()
        data.pop("text")
        return data

    return create_augmented_text(text1, text2, lan1, lan2)


def create_augmented_text_from_aligned_data(data):
    if "text_1" in data:
        lan1 = data.pop("lan_1")
        lan2 = data.pop("lan_2")
        text1 = data.pop("text_1").strip()
        text2 = data.pop("text_2").strip()
        data["text"] = create_augmented_text(text1, text2, lan1, lan2)
    else:
        data["text"] = create_augmented_text(data.pop("text_en"), data.pop("text_fr"), "en", "fr")
    return data


def create_augmented_text(text1, text2, lan1, lan2, separator=None):
    uselanguage_prefix = random.random() < 0.5

    if separator is None:
        separator = random.choice([" ", "   ", "    ", "     ", "\t", "\t\t", "\n", "\n\n"])
    before_lan = ""
    after_lan = random.choice([". ", ": ", " : ", "- ", "-- ", "— ", "\n"])
    brackets = random.choice([""] * 4 + ["[]", "()", "{}", "<>"])
    if brackets:
        before_lan = brackets[0]
        after_lan = after_lan.strip().rstrip(".-—") + brackets[1] + " "
    clear_separation = "\n" in separator or "\t" in separator or bool(brackets)
    how_to_write_language = random.choice(range(4) if clear_separation else range(1, 4))

    text = text1 + separator + text2

    if uselanguage_prefix:
        if how_to_write_language == 0:
            pass
        elif how_to_write_language == 1:
            lan1, lan2 = (LAN_TO_COMPLETE[x][x] for x in [lan1, lan2])
        elif how_to_write_language == 2:
            lan1, lan2 = (LAN_TO_COMPLETE[lan1][x] for x in [lan1, lan2])
        elif how_to_write_language == 3:
            lan1, lan2 = (LAN_TO_COMPLETE[lan2][x] for x in [lan1, lan2])

        if random.random() < 0.5 or not clear_separation:  # Capitalize
            lan1, lan2 = (x.capitalize() for x in [lan1, lan2])

        if random.random() < 0.5:  # Invert
            lan1, lan2 = lan2, lan1
            text1, text2 = text2, text1

        text = before_lan + lan1 + after_lan + text1 + separator + before_lan + lan2 + after_lan + text2

    return text


LAN_TO_COMPLETE = {
    "fr": {
        "fr": "français",
        "en": "anglais",
        "it": "italien",
        "de": "allemand",
        "es": "espagnol",
    },
    "en": {
        "en": "english",
        "fr": "french",
        "it": "italian",
        "de": "german",
        "es": "spanish",
    },
    "it": {
        "fr": "francese",
        "en": "inglese",
        "it": "italiano",
        "de": "tedesco",
        "es": "spagnolo",
    },
    "de": {
        "fr": "französisch",
        "en": "englisch",
        "it": "italienisch",
        "de": "deutsch",
        "es": "spanisch",
    },
    "es": {
        "fr": "francés",
        "en": "inglés",
        "it": "italiano",
        "de": "alemán",
        "es": "español",
    },
}


class DataIteratorClaire(DataIteratorConcat):
    def __init__(self, language="fr", streaming=True, split=None, use_nc=False, subset_regex=None, **kwargs):  # noqa # C901 `...` is too complex
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

        if subset_regex:
            train_files2 = [f for f in train_files if re.search(subset_regex, f, re.IGNORECASE)]
            test_files2 = [f for f in test_files if re.search(subset_regex, f, re.IGNORECASE)]
            assert train_files2, f"No files found in {path}/*/*.txt" + f" with regex {subset_regex} ({train_files})"
            train_files = train_files2
            test_files = test_files2

        assert train_files, f"No files found in {path}/*/*.txt"
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
            if not use_nc:
                if language == "fr" and name not in ["Theatre", "AssembleeNationale", "Senat"]:
                    continue
                elif language == "en" and not any(
                    s in name for s in ["AMI", "ICSI", "Charlotte", "Switchboard", "DialogStudio"]
                ):
                    continue
            dataset_to_filenames.setdefault(name, []).append(filename)

        name = f"Claire:{language.lower()}"
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
                    name=f"{name}:{subname}" + (f":{split}" if split else ""),
                    **kwargs,
                )
                for subname, filenames in dataset_to_filenames.items()
            ],
            name=name,
        )


class DataIteratorEurovoc(DataIteratorParquet):
    def __init__(self, language="en", filter_by_perplexity=True, **kwargs):
        name = f"Eurovoc:{language.lower()}"
        folder = os.path.join(DATA_PATH, "perplexity_corpus_open_llm", f"eurovoc_{language}")
        DataIteratorParquet.__init__(
            self,
            folder,
            name=name,
            filter_fn=filter_by_perplexity_func(1500) if filter_by_perplexity else None,
            postprocess=clean_eurovoc,
            **kwargs,
        )


########################################
# Datasets: French


def filter_by_perplexity_func(threshold):
    return lambda x: len(x["ccnet_perplexity"]) and x["ccnet_perplexity"][0] <= threshold


class DataIteratorGallicaMono(DataIteratorParquet):
    def __init__(self, filter_by_perplexity=True, **kwargs):
        folder = os.path.join(DATA_PATH, "perplexity_corpus_open_llm", "gallica_mono_parquet")
        DataIteratorParquet.__init__(
            self,
            folder,
            key="complete_text",
            name="GallicaMonographies",
            postprocess=html_unescape,  # clean_pdf_extraction_and_html
            filter_fn=filter_by_perplexity_func(815) if filter_by_perplexity else None,
            **kwargs,
        )


class DataIteratorGallicaPress(DataIteratorConcat):
    def __init__(self, filter_by_perplexity=True, **kwargs):
        folder = os.path.join(
            DATA_PATH,
            "perplexity_corpus_open_llm",
        )
        DataIteratorConcat.__init__(
            self,
            [
                DataIteratorParquet(
                    os.path.join(folder, f"gallica_presse_{source}_parquet"),
                    key="complete_text",
                    name=f"GallicaPress:{source}",
                    postprocess=html_unescape,  # clean_pdf_extraction
                    filter_fn=filter_by_perplexity_func(690) if filter_by_perplexity else None,
                    **kwargs,
                )
                for source in ("html", "txt")
            ],
            name="GallicaPress",
        )


class DataIteratorHal(DataIteratorParquet):
    def __init__(self, filter_by_perplexity=True, **kwargs):
        folder = os.path.join(
            DATA_PATH,
            "perplexity_corpus_open_llm",
            "hal_parquet",
        )
        DataIteratorParquet.__init__(
            self,
            folder,
            name="HAL",
            # postprocess=lambda text: clean_pdf_extraction(text, html_escape=True),
            filter_fn=filter_by_perplexity_func(930) if filter_by_perplexity else None,
            **kwargs,
        )


class DataIteratorTheses(DataIteratorParquet):
    def __init__(self, filter_by_perplexity=True, **kwargs):
        folder = os.path.join(
            DATA_PATH,
            "perplexity_corpus_open_llm",
            "theses_parquet",
        )
        DataIteratorParquet.__init__(
            self,
            folder,
            name="Theses",
            # postprocess=lambda text: clean_pdf_extraction(text, html_escape=True),
            filter_fn=filter_by_perplexity_func(2535) if filter_by_perplexity else None,
            key="complete_text",
            **kwargs,
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


class DataIteratorPersee(DataIteratorParquet):
    def __init__(self, **kwargs):
        DataIteratorParquet.__init__(
            self,
            DATA_PATH + "/persee_parquet",
            name="Persee",
            key="complete_text",
            subsample_criteria="file_id",
            # postprocess=lambda text: clean_pdf_extraction(text, html_escape=True),
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
    def __init__(self, regex_parquet=None, **kwargs):
        DataIteratorParquetSplitted.__init__(
            self,
            DATA_PATH + "/other_fr_parquet",
            name="OtherFr" if not regex_parquet else regex_parquet,
            postprocess=kwargs.pop("postprocess", fix_legi),
            regex_parquet=regex_parquet,
            **kwargs,
        )


class DataIteratorOpenDataFr(DataIteratorParquetSplitted):
    def __init__(self, regex_parquet=None, **kwargs):
        DataIteratorParquetSplitted.__init__(
            self,
            DATA_PATH + "/open_data_fr",
            name="OpenData" if not regex_parquet else regex_parquet,
            postprocess=kwargs.pop("postprocess", fix_legi),
            regex_parquet=regex_parquet,
            **kwargs,
        )


########################################
# Datasets: English


class DataIteratorAmericanStories(DataIteratorConcat):
    def __init__(self, streaming=True, from_huggingface=False, filter_by_perplexity=True, **kwargs):
        data_path = os.path.join(DATA_PATH, "perplexity_corpus_open_llm", "americanstories", "*.parquet")

        if from_huggingface is None:
            from_huggingface = not os.path.isdir(data_path)
            logger.info(
                "Using HuggingFace version for AmericanStories"
                if from_huggingface
                else "Using local version for AmericanStories"
            )

        key = "article"

        if not from_huggingface:
            data_files = sorted(glob.glob(data_path))
            assert len(data_files), f"Missing parquet files for {data_path}"
            logger.info(f"Using {len(data_files)} parquet files in {data_path}")

            key = "text"

            # def load_parquet(data_file):
            #     try:
            #         return datasets.load_dataset(
            #             "parquet",
            #             data_files=data_file,
            #             streaming=streaming,
            #             split="train",
            #         )
            #     except Exception as err:
            #         raise RuntimeError(f"Error loading {data_file}") from err

            # datas = {
            #     os.path.splitext(os.path.basename(data_file))[0]: load_parquet(data_file) for data_file in data_files
            # }
            self.parquet_files = data_files
            data_files = {int(os.path.splitext(os.path.basename(data_file))[0]): data_file for data_file in data_files}

            DataIteratorConcat.__init__(
                self,
                [
                    DataIteratorParquet(
                        data_files[year],
                        key=key,
                        subsample_criteria="article_id",
                        name=f"AmericanStories:{year}",
                        # postprocess=remove_simple_lines,
                        filter_fn=filter_by_perplexity_func(2310) if filter_by_perplexity else None,
                        **kwargs,
                    )
                    for year in sorted(data_files.keys())
                ],
                name="AmericanStories",
            )

        else:
            assert not filter_by_perplexity
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
                        # postprocess=remove_simple_lines,
                        filter_fn=filter_by_perplexity_func(2310) if filter_by_perplexity else None,
                        **kwargs,
                    )
                    for year in datas.keys()
                ],
                name="AmericanStories",
            )


class DataIteratorPes2o(DataIteratorConcat):
    def __init__(self, streaming=True, from_huggingface=False, train=None, split_by_type=True, **kwargs):
        name = "PeS2o"

        if from_huggingface is None:
            from_huggingface = not os.path.isdir(f"{DATA_PATH}/peS2o_train_jsonl")
            logger.info(
                "Using HuggingFace version for AmericanStories"
                if from_huggingface
                else "Using local version for AmericanStories"
            )

        if train is not None:
            splits = ["train"] if train else ["validation"]
        else:
            splits = ["validation", "train"]

        if from_huggingface:
            repo = "allenai/peS2o"

            if split_by_type:
                filter_fns = {
                    "s2ag": lambda x: x["source"].startswith("s2ag"),
                    "s2orc": lambda x: not x["source"].startswith("s2ag"),
                }
            else:
                filter_fns = {"": None}

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
                        name=f"{name}:{subset_name+':' if subset_name else ''}{split}",
                        **kwargs,
                    )
                    for split in splits
                    for subset_name, filter_fn in filter_fns.items()
                ],
                name=name,
            )

        else:
            self.json_files = []
            iterators = []
            for split in splits:
                files_regex = f"{DATA_PATH}/peS2o_{split}_jsonl/*.json"
                json_files = glob.glob(files_regex)
                if not len(json_files):
                    raise RuntimeError(f"No json files in {files_regex}")
                logger.info(f"Using {len(json_files)} json files from {files_regex}")
                self.json_files.extend(json_files)
                iterators.append(
                    DataIterator(
                        datasets.load_dataset(
                            "json",
                            streaming=streaming,
                            data_files=json_files,
                            split="train",
                        ),
                        name=f"{name}:{split}",
                        **kwargs,
                    )
                )

            DataIteratorConcat.__init__(self, iterators, name=name)


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
        programming_languages=None,
        max_chars_per_language=None,
        from_huggingface=None,
        **kwargs,
    ):
        if from_huggingface is None:
            from_huggingface = not os.path.isdir(f"{DATA_PATH}/the-stack-dedup")
            logger.info(
                "Using HuggingFace version for the-stack-dedup"
                if from_huggingface
                else "Using local version for the-stack-dedup"
            )

        metadata = json.load(open(_the_stack_metadata_file, encoding="utf8"))
        # Lower case all keys
        metadata = {k.lower(): v for k, v in metadata.items()}

        # Check we are not missing one language
        if programming_languages is None:
            programming_languages = _programming_languages
        for lan in programming_languages:
            assert lan in metadata.keys(), f"Missing language {lan} in metadata file {list(metadata.keys())}"

        dataset_name = "bigcode/the-stack-dedup" if from_huggingface else "parquet"

        DataIteratorConcat.__init__(
            self,
            list(
                self.yield_datasets(
                    programming_languages,
                    metadata=metadata,
                    from_huggingface=from_huggingface,
                    dataset_name=dataset_name,
                    streaming=streaming,
                    max_chars_per_language=max_chars_per_language,
                    **kwargs,
                )
            ),
            name="TheStack",
        )

    def yield_datasets(
        self,
        programming_languages,
        **kwargs,
    ):
        for lan in programming_languages:
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
    import shutil
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
        help="Only dump some examples",
    )
    args = parser.parse_args()

    if args.folder:
        os.makedirs(args.folder, exist_ok=True)
        shutil.copy2(__file__, os.path.join(args.folder, os.path.basename(__file__)))

    def simple_slugify(name):
        return re.sub(r"[ :/]", "--", name).strip("_-")

    def remove_common_prefix(main, sub):
        common_prefix = os.path.commonprefix([main, sub])
        return sub[len(common_prefix) :]

    def test_iterator(  # noqa # C901 `...` is too complex
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
                    print(f"Skipping {name_slug} (already computed)")
                    return stats
                num_billion_words = stats["num words"] / 1_000_000_000
                to_insert = f"{num_billion_words:06.3f}B"
                if "--" in prefix_example_files:
                    prefix_example_files = prefix_example_files.replace("--", "--" + to_insert + "_", 1)
                else:
                    prefix_example_files += "--" + to_insert
        print(f"Computing stats for {name_slug}...")
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
                    f.write(text + "\n")
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

    all_datasets = [get_datasets(name) for name in args.dataset]
    all_datasets = [it for sublist in all_datasets for it in sublist]

    # Early checks to avoid failure in the middle
    for it in all_datasets:
        assert isinstance(it, DataIteratorBase), f"Invalid iterator {it}"
        assert it.name, f"Missing name for {it}"

    for it in all_datasets:
        num_examples = args.num_examples
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

        max_num_examples_per_subset = num_examples  # / len(its)
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