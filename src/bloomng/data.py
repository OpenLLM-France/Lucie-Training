import datasets

import random
import glob
import socket
import os
import hashlib, pickle
import json

_folder = os.path.dirname(os.path.realpath(__file__))
_the_stack_metadata_file = os.path.join(_folder, "assets", "the-stack-dedup-programming-languages-stats.json")
assert os.path.isfile(_the_stack_metadata_file)

DATA_PATH = os.environ.get("DATA_PATH")
if not DATA_PATH:
    for possible_data_path in [
        "/gpfswork/rech/qgz/commun/data/", # Jean-Zay
        "/media/storage0/OpenLLM", # koios
        "/data-storage/OpenLLM", # biggerboi
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

    # Some statistics:
    #
    # Wikipedia.fr:
    # - 2.6M docs
    # - 1.7B words
    # - 9.8B chars
    # Wikipedia.en:
    # - 6.8M docs
    # - 4.7B words
    # - 26.8B chars
    #
    # The Stack -- Python:
    # - 13M docs
    # - 5.4B words
    # - 64B chars
    # The Stack -- Javascript / Java:
    # - 21M / 20M docs
    # - 8.5B / 7.4B words
    # - 142B / 90B chars
    # The Stack -- C++:
    # - 6.3M docs
    #
    # Gallica -- 1st parquet file:
    # - 2k docs
    # - 300M words
    # - 1.8B chars

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
    if download_only: del wikipedia_fr

    wikipedia_en = DataIteratorWikipedia(language="en", **kwargs_wikipedia)
    if download_only: del wikipedia_en

    if download_only: return

    code = DataIteratorCode(**kwargs_code)

    gallica = DataIteratorGallica(**kwargs_gallica)

    if download_only: return
    return (
        f"WikipediaFrEn{factor * 500}kpages-Gallica1G-TheStack{25}Mchars",
        DataIteratorConcat(
            [
                gallica,
                wikipedia_fr,
                wikipedia_en,
                code,
            ]
        )
    )


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
        name="",
    ):
        """
        Args:
            dataset: a dataset object
            key: the key of the text in the dataset
            max_docs: the maximum number of pages to return
            split_around_paragraphs: if True, split the text around paragraphs
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

            # Subsampling
            if self.subsample_rate and self.subsample_rate < 1:
                if self.subsample_criteria == None:
                    r = self.random_generator.random()
                else:
                    criterion = data[self.subsample_criteria]
                    r = string_to_random01(criterion)
                if (r <= self.subsample_rate) if self.subsample_invert else (r > self.subsample_rate):
                    # Skip this example
                    self.idx -= 1
                    return self.__next__()

            text = data[self.key]

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
        if self.split_around_paragraphs or (self.subsample_rate and self.subsample_rate < 1):
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
    normalized_value = (hash_value & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF) / 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF

    return normalized_value

class DataIteratorConcat:
    def __init__(self, datasets):
        self.datasets = datasets
        self.idx = 0

    def __iter__(self):
        return self

    def __len__(self):
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


########################################
# All datasets used

class DataIteratorWikipedia(DataIterator):
    def __init__(
        self,
        language="fr",
        use_latex_version=True,
        streaming=True,
        **kwargs):
        if language == "fr":
            # 1.7B words
            if use_latex_version:
                repo, version = ("OpenLLM-France/wikipedia_latex.fr", "20240101")
            else:
                repo, version = ("OpenLLM-France/wikipedia.fr", "20231220")
        elif language == "en":
            # 4.7B words
            if use_latex_version:
                repo, version = ("OpenLLM-France/wikipedia_latex.en", "20240101")
            else:
                repo, version = ("OpenLLM-France/wikipedia.en", "20240101")
        name = f"{repo}/{version}"
        DataIterator.__init__(
            self,
            datasets.load_dataset(repo, version, streaming=streaming, split="train"),
            subsample_criteria="id",
            name=name,
            **kwargs,
        )


class DataIteratorCode(DataIteratorConcat):
        
    def __init__(
        self,
        streaming=True,
        max_chars_per_language = None,
        **kwargs):

        metadata = json.load(open(_the_stack_metadata_file, "r", encoding="utf8"))  
        # Lower case all keys
        metadata = {k.lower(): v for k, v in metadata.items()}

        languages = [
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

        # Check we are not missing one language
        for lan in languages:
            assert lan in metadata.keys(), f"Missing language {lan} in metadata file {list(metadata.keys())}"

        iterators = []
        for lan in languages:
            data_dir = f"data/{metadata[lan]['name']}"
            it = DataIterator(
                datasets.load_dataset(
                    "bigcode/the-stack-dedup",
                    data_dir=data_dir,
                    streaming=streaming,
                    split="train",
                ),
                key="content",
                subsample_criteria="hexsha",
                max_chars=max_chars_per_language,
                name=f"TheStack_{lan}",
                **kwargs,
            )
            iterators.append(it)

        self.name = "TheStack"

        DataIteratorConcat.__init__(
            self,
            iterators
        )

class DataIteratorGallica(DataIterator):

    def __init__(
        self,
        streaming=True,
        max_parquet_files=None,
        **kwargs):

        parquet_files = glob.glob(DATA_PATH + "/gallica_mono_parquet/*parquet")
        parquet_files = sorted(parquet_files, key=lambda x: int(x.split(".")[-2].split("_")[-1]))
        if max_parquet_files and max_parquet_files < len(parquet_files):
            parquet_files = parquet_files[:max_parquet_files]

        DataIterator.__init__(
            self,
            datasets.load_dataset(
                "parquet",
                data_files = parquet_files,
                streaming=streaming,
                split="train",
            ),
            key="complete_text",
            name="GallicaMonographies",
            postprocess=fix_monographies,
            **kwargs,
        )

def fix_monographies(text):
    if "\nTABLE DES MATIERES" in text:
        text = text.split("\nTABLE DES MATIERES")[0]
    else:
        text = "\n".join(text.split("\n")[:-3])
    text = text.replace("\\", "")
    text = text.replace(" \n", "\n")
    text = text.strip()
    return text

########################################
# Quick test

if __name__ == "__main__":
    
    import tqdm
    import time
    import json

    def test_iterator(it):
        tic = time.time()
        i_page = 0
        num_words = 0
        num_chars = 0
        for i_page, text in enumerate(tqdm.tqdm(it, total=len(it))):
            num_words += len(text.split())
            num_chars += len(text)
        toc = time.time()
        return {
            "time to iterate (sec)": toc - tic,
            "num pages": i_page + 1,
            "num words": num_words,
            "num chars": num_chars,
        }
    
    # NOCOMMIT
    print(json.dumps(test_iterator(DataIteratorGallica()), indent=4))
    exit(0)
    
    # Download all files from HF
    # tokenizer_dataset(download_only=True)

    name, trainset = tokenizer_dataset(train=True)

    print(f"* Tokenizer train set for {name}")
    for subsets in trainset.datasets:
        print(f"  * {subsets.name}")
        stats = test_iterator(subsets)
        print(json.dumps(stats, indent=4))

    _, testset = tokenizer_dataset(train=False)

    print(f"* Tokenizer test set for {name}")
    for subsets in testset.datasets:
        print(f"  * {subsets.name}")
        stats = test_iterator(subsets)
        print(json.dumps(stats, indent=4))

