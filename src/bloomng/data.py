import glob
import hashlib
import json
import os
import pickle
import random
import logging

import datasets
import regex as re
from text import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_folder = os.path.dirname(os.path.realpath(__file__))
_the_stack_metadata_file = os.path.join(
    _folder, "assets", "the-stack-dedup-programming-languages-stats.json"
)
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
        raise RuntimeError(
            "No data path found. You can set it using DATA_PATH environment variable."
        )


########################################
# Custom Iterators


def tokenizer_dataset(
    train=True, streaming=True, debug=False, download_only=False, factor=3
):
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

    name = name.lower()
    if name == "all":
        for name in [
            # Multi-lingual
            "wikipedia",
            "europarl",

            # French
            "gallica_press",
            "gallica_mono",
            "theses",
            "hal",
            "persee",
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

    elif name in ["tok_all", "tok_train"]:
        yield tokenizer_dataset(train=True)
        if name == "tok_all":
            yield tokenizer_dataset(train=True)
    elif name == "tok_test":
        yield tokenizer_dataset(train=False)

    elif name == "code":
        yield ("TheStack", DataIteratorCode(**kwargs))

    elif name == "wikipedia":
        for language in "fr", "en", "de", :
            yield (f"Wikipedia{language.capitalize()}", DataIteratorWikipedia(language=language, **kwargs))
    elif name.startswith("wikipedia_"):
        language = name.split("_")[1].lower()
        yield (f"Wikipedia{language.capitalize()}", DataIteratorWikipedia(language=language, **kwargs))

    elif name == "europarl":
        for language in "fr", "en", "de", "es",:
            yield (f"Europarl{language.capitalize()}", DataIteratorEuroparl(language=language, **kwargs))
    elif name.startswith("wikipedia_"):
        language = name.split("_")[1].lower()
        yield (f"Wikipedia{language.capitalize()}", DataIteratorWikipedia(language=language, **kwargs))

    elif name == "claire":
        yield ("ClaireFr", DataIteratorClaire(language="fr", **kwargs))
        yield ("ClaireEn", DataIteratorClaire(language="en", **kwargs))
    elif name.startswith("claire_"):
        language = name.split("_")[1].lower()
        yield (f"Claire{language.capitalize()}", DataIteratorClaire(language=language, **kwargs))

    # elif name in ["gallica_mono"]:
    #     yield ("GallicaMono", DataIteratorGallicaMono())
    # elif name in ["gallica_press"]:
    #     yield ("GallicaPress", DataIteratorGallicaPress())
    # elif name in ["discours"]:
    #     yield ("DiscoursPublics", DataIteratorDiscoursPublics())
    # elif name in ["american_stories"]:
    #     yield ("AmericanStories", DataIteratorAmericanStories())
    # etc.

    else:
        camel_name = "".join([w.capitalize() for w in name.split("_")])
        python_class = globals().get(f"DataIterator{camel_name}", None)
        if python_class:
            yield (camel_name, python_class(**kwargs))
        else:
            raise RuntimeError(f"Unknown dataset {name}")


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
        split_around_paragraphs=False,
        postprocess=None,
        filter_fn=None,
        name="",
    ):
        """
        Args:
            dataset: a dataset object
            key: the key of the text in the dataset
            max_docs: the maximum number of pages to return
            split_around_paragraphs: if True, split th7e text around paragraphs
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
        self.split_around_paragraphs = split_around_paragraphs
        self.postprocess = postprocess
        self.filter_fn = filter_fn

        self.random_generator = random.Random(42)
        self.idx = 0
        self.texts = []
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
        if self.texts:
            # cf. split_around_paragraphs (remaining texts to serve)
            text = self.texts.pop(0)
        else:
            data = next(self.dataset_iter)

            if self.filter_fn:
                if not self.filter_fn(data):
                    # Skip this example
                    self.idx -= 1
                    return self.__next__()

            # Subsampling
            if self.subsample_rate and self.subsample_rate < 1:
                if self.subsample_criteria == None:
                    r = self.random_generator.random()
                else:
                    criterion = data[self.subsample_criteria]
                    r = string_to_random01(criterion)
                if (
                    (r <= self.subsample_rate)
                    if self.subsample_invert
                    else (r > self.subsample_rate)
                ):
                    # Skip this example
                    self.idx -= 1
                    return self.__next__()

            try:
                text = data[self.key]
            except KeyError:
                raise KeyError(f"Key {self.key} not found in {data.keys()}.")
            except TypeError as err:
                raise TypeError(f"Problem with to process data of type {type(data)} with key {type(self.key)}.") from err

            # Split around paragraphs
            if self.split_around_paragraphs:
                texts = text.split("\n\n")
                if not texts:
                    return self.__next__()
                text = texts[0]
                self.texts = texts[1:]

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
        if self.split_around_paragraphs or (
            self.subsample_rate and self.subsample_rate < 1
        ):
            # Unknown a priori length?
            if self.max_docs:
                return self.max_docs
            if not self.split_around_paragraphs:
                return round(len(self.dataset) * self.subsample_rate)
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
    def __init__(self, data_path, streaming=True, max_parquet_files=None, force_raw=False, **kwargs):
        if not force_raw and os.path.exists(str(data_path) + "_cleaned") and kwargs.get("postprocess", None) is not None:
            data_path = data_path + "_cleaned"
            logger.info(f"Using pre-cleaned in {data_path}")
            kwargs.pop("postprocess", None)
        parquet_files = glob.glob(data_path + "/*parquet")
        if parquet_files and re.match(r".*_\d+\.parquet$", parquet_files[0]):
            parquet_files = sorted(
                parquet_files, key=lambda x: int(x.split(".")[-2].split("_")[-1])
            )
        if max_parquet_files and max_parquet_files < len(parquet_files):
            parquet_files = parquet_files[:max_parquet_files]
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
            parquet_files = sorted(
                parquet_files, key=lambda x: int(x.split(".")[-2].split("_")[-1])
            )
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
    def __init__(self, language="fr", use_latex_version=True, streaming=True, from_huggingface=None, **kwargs):
        version = "20240101"
        repo = (
            f"OpenLLM-France/wikipedia_latex.{language}"
            if use_latex_version
            else f"OpenLLM-France/wikipedia.{language}"
        )
        name = f"{repo}/{version}"

        if from_huggingface is None:
            from_huggingface = not os.path.isdir(f"{DATA_PATH}/wikipedia/{language}")
            logger.info(f"Using HuggingFace version for {name}" if from_huggingface else f"Using local version for {name}")

        if from_huggingface:
            if language == "fr" and not use_latex_version:
                version = "20231220"
            kwargs_dataset = dict(name=version)
        else:
            assert use_latex_version, "Only latex version is available for now"
            repo = "parquet"
            data_files = sorted(glob.glob(f"{DATA_PATH}/wikipedia/{language}/{version}/*.parquet"))
            assert len(data_files), f"Missing parquet files for {DATA_PATH}/wikipedia/{language}/{version}/*.parquet"
            kwargs_dataset = dict(data_files=data_files)

        DataIterator.__init__(
            self,
            datasets.load_dataset(repo, streaming=streaming, split="train", **kwargs_dataset),
            subsample_criteria="id",
            name=name,
            **kwargs,
        )


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
            datasets.load_dataset(
                repo,
                streaming=streaming,
                split="train",
                **kwargs_dataset
            ),
            name=f"Europarl{language.capitalize()}",
            **kwargs,
        )


class DataIteratorClaire(DataIterator):
    def __init__(self, language="fr", streaming=True, split=None, **kwargs):
        path = DATA_PATH + f"/claire_{language}"
        full_files = glob.glob(path + "/*/full.txt")
        train_files = glob.glob(path + "/*/train.txt")
        test_files = glob.glob(path + "/*/test.txt")
        # Ignore full files if we have train files
        full_files = [
            f
            for f in full_files
            if f.replace("full.txt", "train.txt") not in train_files
        ]
        train_files += full_files

        to_exclude = [
            "AntiScam",
            "CaSiNo",
            "CraigslistBargains",
            "FRAMES",
        ]  # JH: highly redundant, short, fake and just dumb
        train_files = [f for f in train_files if not any([e in f for e in to_exclude])]
        test_files = [f for f in test_files if not any([e in f for e in to_exclude])]

        assert train_files, f"No files found in {path}"
        if split is None:
            files = train_files + test_files
        elif split == "train":
            files = train_files
        elif split == "test":
            files = test_files

        DataIterator.__init__(
            self,
            datasets.load_dataset(
                "text",
                data_files={"train": files},
                streaming=streaming,
                sample_by="paragraph",
                split="train",
            ),
            key="text",
            name=f"Claire{language.capitalize()}",
            **kwargs,
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

def clean_pdf_extraction_and_html(text):
    return clean_pdf_extraction(text, html_escape=True)


class DataIteratorGallicaPress(DataIteratorConcat):
    def __init__(self, **kwargs):

        self.name = "GallicaPress"
        DataIteratorConcat.__init__(self, [
            DataIteratorParquet(
                DATA_PATH + f"/gallica_presse_{source}_parquet",
                key="complete_text",
                name=f"GallicaPress:{source}",
                # postprocess=lambda text: clean_pdf_extraction(text, html_escape=True),
                **kwargs,
            ) for source in ("html", "txt")
        ])


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
            logger.info(f"Using HuggingFace version for {self.name}" if from_huggingface else f"Using local version for {self.name}")

        key="article"

        if not from_huggingface:
            data_files = sorted(glob.glob(f"{DATA_PATH}/americanstories/*.parquet"))
            assert len(data_files), f"Missing parquet files for {DATA_PATH}/americanstories/*.parquet"

            key="text"

            datas = dict([
                (
                    os.path.splitext(os.path.basename(data_file))[0], # year
                    datasets.load_dataset(
                        "parquet",
                        data_files=data_file,
                        streaming=streaming,
                        split="train",
                    )
                )
                for data_file in data_files
            ])

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
            splits = ["train", "validation"]

        if split_by_type:
            filter_fns = {
                "s2ag": lambda x: x["source"].startswith("s2ag"),
                "s2orc": lambda x: not x["source"].startswith("s2ag"),
            }
        else:
            filter_fns = {"": None}

        self.name = "Pes2o"
        DataIteratorConcat.__init__(self, [
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
        ])


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
    def __init__(self, streaming=True, max_chars_per_language=None, from_huggingface=None, **kwargs):
        self.name = "TheStack"

        if from_huggingface is None:
            from_huggingface = not os.path.isdir(f"{DATA_PATH}/the-stack-dedup")
            logger.info(f"Using HuggingFace version for {self.name}" if from_huggingface else f"Using local version for {self.name}")

        metadata = json.load(open(_the_stack_metadata_file, encoding="utf8"))
        # Lower case all keys
        metadata = {k.lower(): v for k, v in metadata.items()}

        # Check we are not missing one language
        for lan in _programming_languages:
            assert (
                lan in metadata.keys()
            ), f"Missing language {lan} in metadata file {list(metadata.keys())}"

        dataset_name = "bigcode/the-stack-dedup" if from_huggingface else "parquet"

        DataIteratorConcat.__init__(
            self,
            list(self.yield_datasets(metadata, from_huggingface, dataset_name, streaming, max_chars_per_language, **kwargs))
        )

    def yield_datasets(self, metadata, from_huggingface, dataset_name, streaming, max_chars_per_language, **kwargs):
        for lan in _programming_languages:
            data_dir = f"data/{metadata[lan]['name']}"

            if from_huggingface:
                kwargs_dataset = dict(data_dir=data_dir)
            else:
                data_files = sorted(glob.glob(f"{DATA_PATH}/the-stack-dedup/{data_dir}/*.parquet"))
                assert len(data_files), f"Missing parquet files for {DATA_PATH}/the-stack-dedup/{data_dir}/*.parquet"
                kwargs_dataset = dict(data_files=data_files)

            yield DataIterator(
                datasets.load_dataset(
                    dataset_name,
                    streaming=streaming,
                    split="train",
                    **kwargs_dataset
                ),
                key="content",
                subsample_criteria="hexsha",
                max_chars=max_chars_per_language,
                filter_fn=(lambda x: x["ext"].lower() == "tex") if lan == "tex" else None, # Exclude bbl, bib, ... from LaTeX
                name=f"TheStack:{lan}",
                **kwargs,
            )



########################################
# Quick test

if __name__ == "__main__":
    import tqdm
    import time
    import json
    import argparse
    import random
    random.seed(1234)

    parser = argparse.ArgumentParser(
        description="Test the data iterators and print statistics about datasets.",
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
        help="Folder to dump some example data into",
    )
    parser.add_argument(
        "--ignore_if_exists",
        action="store_true", default=False,
        help="Skip if stat is already computed",
    )
    parser.add_argument(
        "--max_num_examples",
        type=int,
        default=10,
        help="Maximum number of pages to dump as examples (when --folder is specified)",
    )
    parser.add_argument(
        "--only_dump_examples",
        action="store_true", default=False,
        help="To only dump some examples",
    )
    args = parser.parse_args()

    def simple_slugify(name):
        return re.sub(r"[ :/]", "--", name).strip("_-")
    
    def remove_common_prefix(main, sub):
        common_prefix = os.path.commonprefix([main, sub])
        return sub[len(common_prefix):]

    def test_iterator(it, folder=None, name="", ignore_if_exists=False, max_num_examples=0, only_dump_examples=False, prefix_example_files=None):
        name_slug = simple_slugify(name)
        if prefix_example_files is None:
            prefix_example_files = name_slug
        stats = None
        if folder:
            stat_filename = os.path.join(folder, f"stats_{name_slug}.json")
            if os.path.isfile(stat_filename):
                stats = json.load(open(stat_filename, "r", encoding="utf8"))
                if ignore_if_exists and not only_dump_examples:
                    return stats
                num_billion_words = stats['num words'] / 1_000_000_000
                to_insert = f"{num_billion_words:06.3f}B"
                if "--" in prefix_example_files:
                    prefix_example_files = prefix_example_files.replace("--", "--" + to_insert + "_", 1)
                else:
                    prefix_example_files += "--" + to_insert
        tic = time.time()
        i_page = 0
        num_words = 0
        num_chars = 0
        for i_page, text in enumerate(tqdm.tqdm(it, total=len(it))):
            num_words += len(text.split())
            num_chars += len(text)
            if i_page < max_num_examples and folder:
                example_folder = os.path.join(folder, "examples")
                os.makedirs(example_folder, exist_ok=True)
                filename = os.path.join(example_folder, f"{prefix_example_files}")
                if max_num_examples > 1:
                    filename += f"_{i_page:02d}"
                filename += ".txt"
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

    for name, it in get_datasets(args.dataset):
        max_num_examples = args.max_num_examples
        if hasattr(it, "name"):
            name = it.name
        name_slug = simple_slugify(name)
        main_prefix_example_files = None
        main_stat_filename = os.path.join(args.folder, f"stats_{name_slug}.json") if args.folder else None
        if main_stat_filename and os.path.isfile(main_stat_filename) and args.only_dump_examples:
            stats = json.load(open(main_stat_filename, "r", encoding="utf8"))
            num_billion_words = stats['num words'] / 1_000_000_000
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

        max_num_examples_per_subset = max_num_examples / len(its)
        for subset in its:
            subname = subset.name
            max_num_examples = int(max_num_examples_per_subset) + (1 if random.random() < ( max_num_examples_per_subset % 1 ) else 0)
            if max_num_examples == 0 and any([s in subname for s in ( "tex", "python")]):
                max_num_examples = 2
            if "other" in name.lower():
                max_num_examples = args.max_num_examples
            if max_num_examples == 0 and args.only_dump_examples:
                continue
            print(f"* {subname}")
            if main_prefix_example_files:
                prefix_example_files = f"{main_prefix_example_files}{remove_common_prefix(name_slug, simple_slugify(subname))}"
            else:
                prefix_example_files = None
            stats = test_iterator(
                subset,
                folder=args.folder,
                name=subname,
                ignore_if_exists=args.ignore_if_exists,
                max_num_examples=max_num_examples,
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
