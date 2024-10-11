import functools
import glob
import json
import logging
import os
import random
import time
import types
import warnings

import datasets
import numpy as np
import regex as re
import tqdm
from text import (
    canonical_url,
    check_language,
    clean_discours,
    clean_eurovoc,
    clean_gutenberg,
    clean_theses,
    clean_wikipedia,
    fix_legi,
    fix_legi_and_remove_title,
    html_unescape,
    is_obscene,
    is_url_duplicated,
    string_to_random01,
)

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)

_folder = os.path.dirname(os.path.realpath(__file__))
_asset_folder = os.path.join(os.path.dirname(_folder), "assets")
_the_stack_metadata_file = os.path.join(
    _asset_folder, "programming-languages", "the-stack-dedup-programming-languages-stats.json"
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


def get_datasets(name, use_nc=True, scope=None, **kwargs):
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
            for ds in get_datasets(n, use_nc=use_nc, scope=scope, **kwargs):
                yield ds
        return

    multilang_corpora_in_training = [
        "wikipedia",
        "wikiother",
        "gutenberg",
        "europarl",
        "claire",
        "eurovoc",
        "validated_youtube",
        "red_pajama",
    ]
    multilang_corpora = multilang_corpora_in_training + [
        # Discarded from "all" (training dataset)
        "cultura_x",
        "subscene",
        "youtube",
    ]

    name = name.lower()

    if name.startswith("claire"):
        kwargs["use_nc"] = use_nc

    if name == "all":
        for name in (
            multilang_corpora_in_training
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
                "fine_web_edu",
                # Multi-language (with switching)
                "europarl_aligned",
                "croissant_aligned",
                # Code, tex, math...
                "math_pile",
                "code",
            ]
        ):
            if "use_nc" in kwargs:
                use_nc = kwargs.pop("use_nc")
            for ds in get_datasets(name, use_nc=use_nc, scope=scope, **kwargs):
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
            "europarl": ["fr", "en", "de", "es"],
            "eurovoc": ["en", "de", "es", "it"],
            "validated_youtube": ["fr"],
            # "wikipedia": ["fr", "en", "de", "es", "it"],
            # "gutenberg": ["fr", "en", "de", "es", "it"],
            # "cultura_x": ["fr", "en", "de", "es", "it"],
            "red_pajama": ["fr", "de", "es", "it"],
        }.get(name, ["fr", "en", "de", "es", "it"])
        if "use_nc" in kwargs:
            use_nc = kwargs.pop("use_nc")
        for language in languages:
            for ds in get_datasets(f"{name}_{language}", use_nc=use_nc, scope=scope, **kwargs):
                yield ds

    else:
        has_language = any(name.startswith(c + "_") for c in multilang_corpora) and not name.endswith("_aligned")
        if has_language:
            fields = name.split("_")
            language = fields[-1]
            name = "_".join(fields[:-1])
            kwargs["language"] = language
        camel_name = "".join([w.capitalize() for w in name.split("_")])
        class_name = f"DataIterator{camel_name}"
        if scope is None:
            scope = globals()
        python_class = scope.get(f"DataIterator{camel_name}", None)
        if not python_class:
            candidates = sorted([k for k in scope.keys() if k.startswith("DataIterator")])
            raise RuntimeError(f"Cannot find python class {class_name} in {candidates}")
        yield python_class(**kwargs)


def decompose_datasets(dataset, parquet_level=False, return_json_file_if_possible=False):
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
        max_parquet_files=None,
        force_include_all_metadata=None,
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

            max_parquet_files: ignored
            force_include_all_metadata: ignored
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

        # Options when getting metadata
        self.key_init = key
        self.do_uniformize_metadata = False
        self.extra_metadata = {}

        self.random_generator = random.Random(42)
        self.idx = -1
        self.idx_orig = -1
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

    def SetYieldMetadata(self, doit=True, uniformize_metadata=False, extra_metadata=None, id_func=None):
        if doit:
            self.key = None
        else:
            self.key = self.key_init
        self.do_uniformize_metadata = uniformize_metadata
        if extra_metadata:
            self.extra_metadata = extra_metadata
        else:
            self.extra_metadata = {}
        self.id_func = id_func

    def uniformize_metadata(self, data):
        """
        Uniformize metadata for sub-datasets (so that we can make a consistent union)
        data: a dictionary
        """

        # Note :
        # - in the following, the order matters
        # - Existing keys won't be replaced (i.e. renaming is not done in cas of conflict)
        _fields_to_rename = {
            # vvv "id"
            "file_id": "id",
            "article_id": "id",
            "doc_id": "id",
            "hexsha": "id",
            "idx": "id",
            "idx_row": "id",
            # vvv "title"
            "page_title": "title",
            # vvv "date"
            "releasedate": "date",
            "created": "date",
            "added": "date",
            "date_download": "date",
            # vvv "language"
            "lang": "language",
            # vvv "path"
            "file_path": "path",
            "page_path": "path",
        }

        _fields_to_regroup_under = {
            "hexsha": "id",
            r".*id": "id",
            r"idx.*": "id",  # idx, idx_row
            # "__index_level_0__":  "id", # Pandas index
            "author": "author",
            "authoryearofbirth": "author",
            "authoryearofdeath": "author",
            r"ccnet_.*": "quality_signals",
            r"rps_.*": "quality_signals",
            r"fasttext_.*": "quality_signals",
            r"chunk.*": "quality_signals",
            r".*count": "quality_signals",
            r".*score": "quality_signals",
            r".*ratio": "quality_signals",
            r".*fraction": "quality_signals",
            r".*length": "quality_signals",
            "ocr": "quality_signals",  # Gallica
            "contain_at_least_two_stop_words": "quality_signals",  # MathPile
            "char_num_after_normalized": "quality_signals",  # MathPile
            r"page.*": "extra",  # page, page_index
            r".*name": "extra",  # book_name, newspaper_name, ...
            r".*path": "extra",  # file_path, page_path, ...
            r".+date": "extra",
            r".*time": "extra",  # max_forks_repo_forks_event_max_datetime (in TheStack), ...
            r".*rights": "extra",  # usagerights, ...
            "version": "extra",  # in PeS2o
            "edition": "extra",  # in AmeridanStories
            "dump": "extra",  # in FineWebEdu
            "subset": "extra",  # in MathPile
            "sujet": "extra",  # in adandements.fr
            "sort": "extra",  # in adandements.fr
            "expose": "extra",  # in adandements.fr
            "loi": "extra",  # in adandements.fr
            "texteloi_id": "extra",  # in adandements.fr
            "intervenants": "extra",  # in adandements.fr
            "signataires": "extra",  # in adandements.fr
            "type": "extra",  # in MathPile
            "mimetype": "extra",  # in MathPile
            "ext": "extra",  # in TheStack (file extension)
            "partition": "extra",  # in RedPajama (it gives the subset)
            "source_domain": "extra",  # in RedPajama (should be included in the url)
            "headline": "extra",
            "size": "extra",  # in TheStack
            "added": "extra",
            "dataset": "extra",
            "digest": "extra",
            "byline": "extra",
            "question": "extra",  # in MathPile (a dict)
            "answers": "extra",  # in MathPile (a list)
        }
        _fields_to_regroup_under_exact = {k: v for k, v in _fields_to_regroup_under.items() if "." not in k}
        _fields_to_regroup_under_fuzzy = {k: v for k, v in _fields_to_regroup_under.items() if "." in k}

        _fields_prefixes_to_remove = [
            # Useless stuff in TheStack
            "max_issues_repo_",
            "max_forks_repo_",
            "max_stars_repo_",
        ]
        _fields_suffixes_to_remove = []
        _fields_to_remove = [
            # vvv - Info that became out of context
            "complete_text",
            "is_duplicate",
            "__index_level_0__",  # Pandas index
            # # vvv - Info that is not useful (index in the original dataset, when there are some unique IDs somewhere)
            # "idx_row",  # in HAL, Theses, ...
            # "idx",
            # "__index_level_0__",  # in CroissantAligned
            # # vvv - Info that is maybe non consistent after text processing (and can be recomputed easily)
            # "word_count",
            # "character_count",
            # "token_count",
            # "page_count",
            # # vvv - Too specific info
            # "question",  # This is a dictionary, in MathPile
            # "answers",  # This is a list, in MathPile
        ]

        is_programming_language = "hexsha" in data and "ext" in data

        self.conform_metadata(data)

        # Uniformize field names : rename some keys (done in the order of renaming)
        for old_key, new_key in _fields_to_rename.items():
            if old_key in data and (
                new_key not in data
                or (new_key == "id" and old_key == "doc_id")  # RedPajama : id was set to an internal file path
            ):
                data[new_key] = data.pop(old_key)

        # Enforce types of raw data
        self.enforce_types(data)

        # - Special thing to get urls
        if data.get("id", "").startswith("http://") and "url" not in data:
            data["url"] = data["id"]

        # Set some values under other meta-fields
        others = {}
        for k in list(data.keys()):
            do_match_exactly = k in _fields_to_regroup_under_exact
            do_match_fuzzy = False
            if not do_match_exactly:
                for pattern, new_field in _fields_to_regroup_under_fuzzy.items():
                    if re.match(pattern, k):
                        do_match_fuzzy = new_field
                        break
            if do_match_exactly or do_match_fuzzy:
                v = data[k]
                new_field = _fields_to_regroup_under[k] if do_match_exactly else do_match_fuzzy
                others[new_field] = others.get(new_field, {}) | {k: v}
                del data[k]
        if others:
            for k, v in others.items():
                # avoid "id" : {"id": "value"}
                if len(v) == 1 and list(v.keys())[0] == k:
                    v = list(v.values())[0]
                data[k] = v

        # Remove some keys
        for key in list(data.keys()):
            if (
                key in _fields_to_remove
                or key.endswith(tuple(_fields_suffixes_to_remove))
                or key.startswith(tuple(_fields_prefixes_to_remove))
            ):
                del data[key]

        # Add extra metadata (for the whole dataset)
        if self.extra_metadata:
            for k, v in self.extra_metadata.items():
                if k not in data:
                    data[k] = v
                elif k == "source":
                    assert isinstance(data[k], str) and isinstance(v, str), f"Cannot merge {k}={data[k]} and {v}"
                    data[k] = v + "/" + data[k]
                elif k in ["language"]:
                    # Let the original value
                    pass
                else:
                    raise NotImplementedError(f"Warning: {k} already in metadata. Combination not implemented.")

        # - Special stuff for languages
        if is_programming_language and "language" in data:
            # Programming languages (not natural)
            lang = data["language"]
            data["language"] += f"programming:{lang}"
        if "languages" in data:
            assert (
                not is_programming_language
            ), "Not Implemented : mixing programming language with natural language information"
            data["language"] = json.dumps(data.pop("languages"), ensure_ascii=False)

        # - Special stuff: author infos -> author
        if "authoryearofbirth" in data or "authoryearofdeath" in data:
            data["author"] = {
                "name": data.get("author"),
                "yearofbirth": data.get("authoryearofbirth"),
                "yearofdeath": data.get("authoryearofdeath"),
            }

        if self.id_func and "id" not in data:
            data["id"] = self.id_func(data, self.idx_orig, self.idx)

        # At last, enforce types of final data, avoiding to have embedded dictionaries
        self.enforce_types(data, no_dict=True)

    @staticmethod
    def conform_metadata(data, flatten="all"):
        # Convert json / Flatten meta
        for metafieldname in "metadata", "meta", "quality_signals":
            do_flatten = (flatten == "all") or (flatten and metafieldname in flatten)
            if metafieldname in data:
                meta = data[metafieldname]
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except json.JSONDecodeError:
                        try:
                            # meta = json.loads(meta.replace("'", '"'))
                            meta = eval(meta)
                        except Exception:
                            pass
                if isinstance(meta, dict):
                    if do_flatten:
                        data.update(meta)
                        del data[metafieldname]
                    else:
                        data[metafieldname] = meta

    @staticmethod
    def enforce_types(data, no_dict=False):
        # Enforce types for some values
        _enforced_types = {
            "id": str,  # can be an int also
            "date": str,  # can be datetime also
            "page": str,  # can be int also
            "author": str,  # Fix author="None" (Gallica)
            "authoryearofbirth": int,
            "authoryearofdeath": int,
            # "quality_signals": str,  # with no_dict=True to use json.dumps(...)
            # "extra": str,  # with no_dict=True to use json.dumps(...)
        }

        for key, val in data.items():
            target_type = _enforced_types.get(key)
            if no_dict and isinstance(val, dict):
                target_type = str
            if target_type and val is not None:
                if not isinstance(val, target_type):
                    if (target_type == str) and isinstance(val, (dict, list)):
                        val = json.dumps(val, ensure_ascii=False)
                    else:
                        try:
                            val = target_type(val)
                        except Exception as err:
                            raise ValueError(f"Cannot convert {key}={val} to {target_type}") from err
                    data[key] = val

                elif val == "None":  # and target_type == str (from condition above)
                    data[key] = None

    def __iter__(self):
        self.idx = -1
        self.idx_orig = -1
        return self

    def _get_next(self):
        self.idx_orig += 1
        try:
            data = next(self.dataset_iter)
            if self.preprocess:
                data = self.preprocess(data)
            return data
        except TypeError as err:
            # Sometimes this can occur because of empty transcription:
            # TypeError: Couldn't cast array of type binary to null
            warnings.warn(f"Got an exception {err}", stacklevel=2)
            return self._get_next()

    def __next__(self):
        self.idx += 1
        if self.max_docs and self.idx >= self.max_docs:
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

        if self.key_init not in data and "text" in data:
            self.key_init = "text"

        text_key = self.key if self.key else self.key_init
        try:
            text = data[text_key]
        except KeyError:
            raise KeyError(f"Key '{text_key}' not found in {data.keys()}.") from None

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

        if not self.key:
            # Normalize text key
            data["text"] = text
            if self.key_init != "text" and self.key_init in data:
                del data[self.key_init]

            if self.do_uniformize_metadata:
                self.uniformize_metadata(data)
            else:
                # Minimal conversion
                self.conform_metadata(data, flatten=None)  # flatten="metadata")

            return data

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


class DataIteratorCustom(DataIterator):
    def __init__(
        self,
        text_data=None,
        repeat=8,
        insert_special=True,
        name="Custom",
        **kwargs,
    ):
        if text_data is None:
            text_data = [
                """
- Bonjour Lucie, comment vas-tu ?
- Et bien moi, ça va très bien, merci. Et toi ?""",
            ]

        if insert_special is True:
            insert_special = [
                (
                    2,
                    """\
    A. Ils parte en vacances en Aout.\n\
    B. Ils pars en vacances en Aout.\n\
    C. Ils partent en vacances en Aout.\n\
    D. Ils partente en vacances en Aout.\n\
    """,
                )
            ]

        if repeat:
            text_data = [text for text in text_data for _ in range(repeat)]

        for idx, text in insert_special:
            assert idx < len(text_data)
            text_data[idx] = text

        DataIterator.__init__(
            self,
            datasets.Dataset.from_dict(
                {"text": text_data},
                split="train",
            ),
            name=name,
            **kwargs,
        )


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

        def fix_wiki_links(data, source="wiktionary"):
            # TODO: fix that in the original datasets
            # (Wiktionary.fr and Wikisource.fr by OpenLLM-France on Hugging Face)
            data["url"] = data["url"].replace("fr.wikipedia.org", f"fr.{source}.org").replace("--", "/")
            return data

        def fix_wiki_links_func(source):
            return lambda x: fix_wiki_links(x, source=source)

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
                    preprocess=fix_wiki_links_func(subname),
                    postprocess=postprocess,
                    subsample_criteria="id",
                    **kwargs,
                )
                for subname, filenames in parquet_per_subfolder.items()
            ],
            name=name,
        )


class DataIteratorGutenberg(DataIteratorParquet):
    def __init__(
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
            postprocess=lambda text: clean_gutenberg(text),
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


def analyze_bilingual_french_english_data(data, add_language_in_data=False):
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

    text, lan1, lan2 = create_augmented_text(text1, text2, lan1, lan2)
    return text


def create_augmented_text_from_aligned_data(data):
    if "text_1" in data:
        lan1 = data.pop("lan_1")
        lan2 = data.pop("lan_2")
        text1 = data.pop("text_1").strip()
        text2 = data.pop("text_2").strip()
        data["text"], lan1, lan2 = create_augmented_text(text1, text2, lan1, lan2)
    else:
        data["text"], lan1, lan2 = create_augmented_text(data.pop("text_en"), data.pop("text_fr"), "en", "fr")
    data["languages"] = [lan1, lan2]
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

    return text, lan1, lan2


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
    def __init__(self, language="fr", streaming=True, split=None, use_nc=False, subset_regex=None, **kwargs):
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


class DataIteratorStac(DataIterator):
    def __init__(self, streaming=True, **kwargs):
        filenames = [DATA_PATH + f"/stac/{subset}.txt" for subset in ["train", "test"]]
        DataIterator.__init__(
            self,
            datasets.load_dataset(
                "text",
                data_files={"train": filenames},
                streaming=streaming,
                sample_by="paragraph",
                split="train",
            ),
            name="Stac",
            **kwargs,
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


class DataIteratorYoutube(DataIteratorParquet):
    def __init__(self, language="en", filter_by_perplexity=True, **kwargs):
        name = f"YouTube:{language.lower()}"
        folder = os.path.join(DATA_PATH, "YouTube", language)
        DataIteratorParquet.__init__(
            self,
            folder,
            name=name,
            **kwargs,
        )


class DataIteratorValidatedYoutube(DataIterator):
    def __init__(self, language="fr", streaming=True, **kwargs):
        path = DATA_PATH + f"/youtube_{language}"
        files = glob.glob(path + "/*.txt")

        assert files, f"No files found in {path}/*.txt"

        name = f"ValidatedYouTube:{language}"
        DataIterator.__init__(
            self,
            datasets.load_dataset(
                "text",
                data_files={"train": files},
                streaming=streaming,
                sample_by="paragraph",
                split="train",
            ),
            name=name,
            **kwargs,
        )


class DataIteratorCulturaX(DataIteratorConcat):
    def __init__(self, language="fr", split="train", streaming=True, **kwargs):
        num_parquets = {
            "it": 256,
            "fr": 512,
            "de": 512,
            "es": 512,
            "en": 3072,
        }.get(language, None)

        assert num_parquets, f"Unsupported language {language}. Number of parquets not defined (512?). Please visit https://huggingface.co/datasets/uonlp/CulturaX/tree/main/{language}"
        num_parquets = min(num_parquets, 512)  # Arbitrary limit for English !

        def filter_fn(data, language, source=None):
            # returns True if the example is to be kept, False otherwise
            if source and (data["source"] != source):
                return False
            if is_url_duplicated(data["url"], language):
                return False
            if is_obscene(data["text"], language):
                return False
            return True

        # # DEBUG
        # def preprocess(data):
        #     data['text'] = '\n'.join([data['url'], data['text']])
        #     return data

        DataIteratorConcat.__init__(
            self,
            [
                DataIterator(
                    datasets.load_dataset(
                        "uonlp/CulturaX",
                        # language,
                        data_files=f"{language}/{language}_part_{iparquet:05d}.parquet",
                        streaming=streaming,
                        split=split,
                        # token=True,
                    ),
                    # name=f"CulturaX:{language.lower()}:{split}:{source.lower()}",
                    # filter_fn=lambda data: filter_fn(data, language, source),
                    name=f"CulturaX:{language.lower()}:{split}_{iparquet}",
                    filter_fn=lambda data: filter_fn(data, language),
                    **kwargs,
                )
                # for source in ('mC4', 'OSCAR-2019', 'OSCAR-2109', 'OSCAR-2201', 'OSCAR-2301')
                for iparquet in tqdm.tqdm(range(num_parquets), desc="Initializing CulturaX...")
            ],
            name=f"CulturaX:{language.lower()}:{split}",
        )


class DataIteratorFineWebEdu(DataIteratorConcat):
    def __init__(self, split="train", target_years=[2024, 2023, 2022, 2021, 2020, 2019], streaming=True, **kwargs):  # noqa # B006
        repo = "HuggingFaceFW/fineweb-edu"
        for folder in [
            "/gpfsscratch/rech/qgz/commun/raw_data/fineweb-edu",
        ]:
            if os.path.isdir(folder):
                repo = folder
                print(f"Using local FineWebEdu data in {repo}")
                break
        builder_configs = datasets.load_dataset_builder(repo).builder_configs
        sources = [k for k in builder_configs.keys() for year in target_years if k.startswith(f"CC-MAIN-{year}")]
        # Set of valid domains
        valid_domains_path = os.path.join(_asset_folder, "urls_robots/valid_domains_fineweb_edu.json")
        with open(valid_domains_path) as fp:
            valid_domains = set(json.load(fp))

        def filter_fineweb_edu(data):
            url = data["url"]
            domain = canonical_url(url)
            if is_url_duplicated(url, "en"):
                return False
            if domain not in valid_domains:
                return False
            return True

        DataIteratorConcat.__init__(
            self,
            [
                DataIterator(
                    datasets.load_dataset(
                        repo,
                        data_files=f"data/{source}/*.parquet",
                        streaming=streaming,
                        split=split,
                    ),
                    name=f"FineWebEdu:{source.lower()}",
                    filter_fn=filter_fineweb_edu,
                    **kwargs,
                )
                for source in sources
            ],
            name="FineWebEdu",
        )


class DataIteratorRedPajama(DataIteratorConcat):
    def __init__(self, language="fr", streaming=True, **kwargs):
        data_path = None
        for path in [
            f"/lustre/fsn1/projects/rech/qgz/uzq54wg/processed_redpajama/v3/pii_removal/{language}",
            f"/data-storage/storage0/corpus_openllm/redpajama/{language}",
        ]:
            if os.path.isdir(path):
                data_path = path
                break
        assert data_path, f"Data path not found for RedPajama in {path}"
        DataIteratorConcat.__init__(
            self,
            [
                DataIterator(
                    datasets.load_dataset(
                        "parquet",
                        data_files={"train": os.path.join(data_path, snapshot, "*.parquet")},
                        streaming=streaming,
                        split="train",
                    ),
                    name=f"RedPajama:{language.lower()}:{snapshot.lower()}",
                    **kwargs,
                )
                for snapshot in os.listdir(data_path)
            ],
            name=f"RedPajama:{language.lower()}",
        )


class CheckedDataIterator(DataIterator):
    def __init__(self, *kargs, **kwargs):
        DataIterator.__init__(self, *kargs, **kwargs)
        self.first = DataIterator.__next__(self)
        assert self.first is not None

    def __next__(self):
        if self.first is None:
            return DataIterator.__next__(self)
        data = self.first
        self.first = None
        return data


########################################
# Datasets: French


def filter_by_perplexity_func(threshold):
    return lambda x: filter_by_perplexity(x, threshold)


def filter_by_perplexity(x, threshold):
    perplexities = x["ccnet_perplexity"]
    if not perplexities:
        return False
    # first_is_lower = perplexities and perplexities[0] <= threshold
    mean_is_lower = np.mean(perplexities) <= threshold
    # median_is_lower = np.median(perplexities) <= threshold
    mean_or_median_is_lower = mean_is_lower or np.median(perplexities) <= threshold
    return mean_or_median_is_lower


def repare_overlapping_chunks(list_of_chunks):
    previous_end = None
    list_of_chunks_ = []
    for start, end in list_of_chunks:
        if (previous_end is not None) and (previous_end > start):
            start = previous_end
        list_of_chunks_.append((start, end))
        previous_end = end
    return list_of_chunks_


def preproc_gallica(data):
    text = data["complete_text"]

    L = 10000
    list_of_chunks = []  # list of chunk start and end
    good_ppl = [
        (ppl <= 1000) and (ppl >= 10) and (lan == "fr") and (lan_score >= 0.65)
        for ppl, lan, lan_score in zip(
            data["ccnet_perplexity"], data["fasttext_language"], data["ccnet_language_score"]
        )
    ]
    for i, actual_chunk_is_good in enumerate(good_ppl):
        if actual_chunk_is_good:
            start = text.rfind("\n", 0, i * L)
            start = start if start != -1 else 0
            end = text.find("\n", (i + 1) * L)
            end = end if end != -1 else len(text)
            list_of_chunks.append((start, end))

    list_of_chunks = repare_overlapping_chunks(list_of_chunks)

    cleaned_text = ""
    previous_end = None
    for start, end in list_of_chunks:
        if previous_end is None and start > 0:
            cleaned_text += "\n\n[...]\n\n"
        if previous_end is not None and start > previous_end:
            cleaned_text += "\n\n[...]\n\n"
        cleaned_text += text[start:end]
        previous_end = end

    cleaned_text = "" if cleaned_text == "\n\n[...]\n\n" else cleaned_text  # In the case all the text is removed
    data["complete_text"] = cleaned_text
    return data


class DataIteratorGallicaMono(DataIteratorParquet):
    def __init__(self, filter_by_perplexity=True, **kwargs):
        folder = os.path.join(DATA_PATH, "perplexity_corpus_open_llm", "gallica_mono_parquet")
        DataIteratorParquet.__init__(
            self,
            folder,
            key="complete_text",
            name="GallicaMonographies",
            preprocess=preproc_gallica,
            postprocess=html_unescape,  # clean_pdf_extraction_and_html
            # filter_fn=filter_by_perplexity_func(815) if filter_by_perplexity else None,
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
                    preprocess=preproc_gallica,
                    postprocess=html_unescape,  # clean_pdf_extraction
                    # filter_fn=filter_by_perplexity_func(690) if filter_by_perplexity else None,
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


def preproc_theses(data, threshold=2000):
    complete_text = data["complete_text"]
    filtered_text = ""

    for idx, (start, end, avg_logprob, lan) in enumerate(
        zip(
            data["chunk_start"],
            data["chunk_end"],
            data["ccnet_avg_log_prob"],
            # data["ccnet_language_score"],
            data["fasttext_language"],
        )
    ):
        chunk = complete_text[start:end]
        if idx <= 1:
            filtered_text += chunk
        elif 10**avg_logprob > threshold:
            pass
        elif lan not in ["fr", "en", "it", "es", "de"]:
            pass
        else:
            filtered_text += chunk
    cleaned_text = clean_theses(filtered_text)
    data["text"] = cleaned_text
    data["word_count"] = len(cleaned_text.split())
    data["character_count"] = len(cleaned_text)
    return data


def filter_thesis_heuristic(data):
    if data["word_count"] < 1000:
        return False
    if data["character_count"] < 10000:
        return False
    return True


class DataIteratorTheses(DataIteratorParquet):
    def __init__(self, filter_by_perplexity=True, **kwargs):
        folder = os.path.join(
            DATA_PATH,
            "perplexity_corpus_open_llm_v2",
            "theses_parquet",
        )
        DataIteratorParquet.__init__(
            self,
            folder,
            name="Theses",
            preprocess=preproc_theses,
            # postprocess=clean_theses, # clean_theses is called in preproc_theses
            filter_fn=filter_thesis_heuristic,
            key="text",
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


class DataIteratorSubscene(DataIterator):
    def __init__(self, split="train", language="fr", streaming=True, **kwargs):
        file = f"{DATA_PATH}/subscene/{split}/corpus.jsonl"
        assert os.path.isfile(file), f"Missing file {file}"

        key = {
            "fr": "french",
            "en": "english",
            "de": "german",
            "es": "spanish",
            "it": "italian",
        }.get(language)
        assert key is not None, f"Unsupported language {language}"

        NO_TEXT = {"text": ""}

        def preprocess(x):
            text = x.get(key)
            if text is None:
                return NO_TEXT
            text = text.get("subtitles")
            if text is None:
                print(f"Missing subtitles in {text.keys()}")
                return NO_TEXT
            assert isinstance(text, list)
            return {"text": "\n".join(text)}

        DataIterator.__init__(
            self,
            datasets.load_dataset(
                "json",
                data_files=file,
                streaming=streaming,
                split="train",
            ),
            name=f"Subscene:{language}",
            preprocess=preprocess,
            **kwargs,
        )


########################################
# Datasets: English


class DataIteratorAmericanStories(DataIteratorConcat):
    def __init__(
        self, streaming=True, from_huggingface=False, filter_by_perplexity=True, max_parquet_files=None, **kwargs
    ):
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

            if max_parquet_files:
                data_files = {
                    k: v for i, (k, v) in enumerate(sorted(data_files.items(), reverse=True)) if i < max_parquet_files
                }

            logger.info(f"Using {len(data_files)} parquet files in {data_path}")

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
                        max_parquet_files=max_parquet_files,
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

            if max_parquet_files:
                datas = {k: v for i, (k, v) in enumerate(sorted(datas.items(), reverse=True)) if i < max_parquet_files}

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
    def __init__(
        self,
        streaming=True,
        from_huggingface=False,
        train=None,
        split_by_type=True,
        force_include_all_metadata=False,
        **kwargs,
    ):
        name = "PeS2o"

        if from_huggingface is None:
            from_huggingface = not os.path.isdir(f"{DATA_PATH}/peS2o_train_jsonl")
            logger.info(
                "Using HuggingFace version for AmericanStories"
                if from_huggingface
                else "Using local version for AmericanStories"
            )
        if force_include_all_metadata:
            from_huggingface = True
            split_by_type = False

        if train is not None:
            splits = ["train"] if train else ["validation"]
        else:
            splits = ["validation", "train"]

        if from_huggingface:
            repo = "allenai/peS2o"

            if split_by_type:
                filter_fns = {
                    "s2orc": lambda x: not x["source"].startswith("s2ag"),
                    "s2ag": lambda x: x["source"].startswith("s2ag"),
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
                            trust_remote_code=True,
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
                if kwargs.get("max_parquet_files"):
                    json_files = json_files[: kwargs["max_parquet_files"]]
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


class DataIteratorPile(DataIteratorConcat):
    def __init__(self, streaming=True, train=True, **kwargs):
        if train is not None:
            splits = ["train"] if train else ["val"]
        else:
            splits = ["train", "val", "test"]

        name = "Pile"

        iterators = []
        parent_folder = f"{DATA_PATH}/pile-uncopyrighted"
        # train_regex = f"{parent_folder}/train/*.jsonl.zst"
        train_regex = f"{parent_folder}/train_sorted_undup/*.jsonl"
        # self.json_files = []
        for type in splits:
            is_train = type == "train"
            files_regex = train_regex if is_train else f"{parent_folder}/{type}.jsonl.zst"
            type = type.replace("pile_", "")
            json_files = glob.glob(files_regex)
            if not len(json_files):
                raise RuntimeError(f"No json files in {files_regex}")
            logger.info(f"Using {len(json_files)} json files from {files_regex}")
            # self.json_files.extend(json_files)
            for json_file in sorted(json_files):
                iterators.append(
                    DataIterator(
                        datasets.load_dataset(
                            "json",
                            streaming=streaming,
                            data_files=json_file,
                            split="train",
                        ),
                        name=name
                        + (
                            f":{os.path.basename(json_file).split('.')[0].replace('pile_', '')}"
                            if is_train
                            else f":{type}"
                        ),
                        **kwargs,
                    )
                )

        DataIteratorConcat.__init__(self, iterators, name=name)


class DataIteratorMonologyPile(DataIteratorConcat):
    def __init__(self, streaming=True, train=True, **kwargs):
        repo = "monology/pile-uncopyrighted"

        json_files = [f"train/{i:02d}.jsonl.zst" for i in [11, 13, 25]]
        name = "MonologyPile"

        DataIteratorConcat.__init__(
            self,
            [
                DataIterator(
                    datasets.load_dataset(
                        repo,
                        streaming=streaming,
                        data_files=json_file,
                        split="train",
                    ),
                    name=f"{name}:{os.path.basename(json_file).split('.')[0]}",
                    **kwargs,
                )
                for json_file in json_files
            ],
            name=name,
        )


class DataIteratorMathPile(DataIteratorConcat):
    def __init__(self, streaming=True, train=True, **kwargs):
        if train is not None:
            splits = ["train"] if train else ["validation"]
        else:
            splits = ["validation", "train"]

        name = "MathPile"

        train_iterators = []
        valid_iterators = []
        # self.json_files = []
        for split in splits:
            is_train = split == "train"
            split_folder = f"{DATA_PATH}/mathpile_commercial/{split}"
            for type in sorted(os.listdir(split_folder)):
                preprocess = None
                if type == "stackexchange":

                    def preprocess(x):
                        question = x["question"]["Body"]
                        answers = [a["Body"] for a in x["answers"]]
                        return x | {"text": question + "\n\n".join(answers)}

                files_regex = f"{split_folder}/{type}/*.jsonl"
                json_files = glob.glob(files_regex)
                if not len(json_files):
                    raise RuntimeError(f"No json files in {files_regex}")
                logger.info(f"Using {len(json_files)} json files from {files_regex}")
                # self.json_files.extend(json_files)
                (train_iterators if is_train else valid_iterators).append(
                    DataIterator(
                        datasets.load_dataset(
                            "json",
                            streaming=streaming,
                            data_files=json_files,
                            split="train",
                        ),
                        preprocess=preprocess,
                        name=f"{name}:{type}" if is_train else f"{name}:{split}:{type}",
                        **kwargs,
                    )
                )

        DataIteratorConcat.__init__(
            self, train_iterators + [DataIteratorConcat(valid_iterators, name=f"{name}:validation")], name=name
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
            if kwargs.get("max_parquet_files"):
                data_files = data_files[: kwargs["max_parquet_files"]]
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
# Test Helpers


def test_iterator(
    it,
    folder=None,
    name="",
    ignore_if_exists=False,
    num_examples=0,
    only_dump_examples=False,
    prefix_example_files=None,
    max_examples=None,
    long_examples=False,
):
    name_slug = simple_slugify(name)
    if prefix_example_files is None:
        prefix_example_files = name_slug
    stats = None
    if folder:
        stat_filename = os.path.join(folder, f"stats_{name_slug}.json")
        if os.path.isfile(stat_filename):
            stats = json.load(open(stat_filename, encoding="utf8"))
            if len(stats):
                if ignore_if_exists and not only_dump_examples:
                    print(f"Skipping {name_slug} (already computed)")
                    return stats
                # num_billion_words = stats["num words"] / 1_000_000_000
                # to_insert = f"{num_billion_words:06.3f}B"
                # if "--" in prefix_example_files:
                #     prefix_example_files = prefix_example_files.replace("--", "--" + to_insert + "_", 1)
                # else:
                #     prefix_example_files += "--" + to_insert
        elif ignore_if_exists:
            # Create an empty file to avoid recomputing
            json.dump({}, open(stat_filename, "w", encoding="utf8"))
    print(f"Computing stats for {name_slug}...")
    tic = time.time()
    num_docs = 0
    num_words = None
    num_chars = None
    num_dumped = 0
    for text in tqdm.tqdm(it, total=len(it)):
        if max_examples and num_dumped >= max_examples:
            break
        num_docs += 1

        # Accumulate number of words and characters
        if isinstance(text, str):
            if num_words is None:
                num_words = 0
                num_chars = 0
            nw = len(text.split())
            num_words += nw
            num_chars += len(text)
        else:
            assert isinstance(text, dict)
            if num_words is None:
                num_words = {}
                num_chars = {}
            nw = 0
            for k, v in text.items():
                if isinstance(v, list):
                    v = " ".join(v)
                assert isinstance(v, str), f"Invalid type for {k}: {v}"
                if k not in num_words:
                    num_words[k] = 0
                    num_chars[k] = 0
                nwi = len(v.split())
                nw += nwi
                num_words[k] += nwi
                num_chars[k] += len(v)

        if num_dumped < num_examples and folder and (not long_examples or nw > 50_000):
            example_folder = os.path.join(folder, "long_examples" if long_examples else "examples")
            os.makedirs(example_folder, exist_ok=True)
            filename = os.path.join(example_folder, f"{prefix_example_files}")
            if num_examples > 1:
                filename += f"_{num_dumped:02d}"
            filename += ".txt"
            if num_dumped == 0:
                print(f"Dumping {filename}")
            with open(filename, "w", encoding="utf8") as f:
                f.write(text + "\n")
            num_dumped += 1
        elif num_dumped >= num_examples and only_dump_examples:
            break
    if only_dump_examples:
        return {}
    if num_docs <= 0:
        raise RuntimeError("No page found, or iterations stopped before completion (stats are not full)")
    toc = time.time()
    stats = {
        "time to iterate (sec)": toc - tic,
        "num pages": num_docs,
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


def simple_slugify(name):
    return re.sub(r"[ :/]", "--", name).strip("_-")


########################################
# Main

if __name__ == "__main__":
    import argparse
    import shutil

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
        "--folder",
        type=str,
        default=os.path.join(_asset_folder, "stats_raw"),
        # default=None,
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
        "--max_examples",
        type=int,
        default=None,
        help="Maximum number of samples to iterate on",
    )
    parser.add_argument(
        "--only_dump_examples",
        action="store_true",
        default=False,
        help="Only dump some examples",
    )
    parser.add_argument(
        "--long_examples",
        action="store_true",
        default=False,
        help="Only dump long examples (more than 50k words)",
    )
    args = parser.parse_args()

    if args.folder:
        os.makedirs(args.folder, exist_ok=True)
        shutil.copy2(__file__, os.path.join(args.folder, os.path.basename(__file__)))

    def remove_common_prefix(main, sub):
        common_prefix = os.path.commonprefix([main, sub])
        return sub[len(common_prefix) :]

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

        try:
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
                    max_examples=args.max_examples,
                    long_examples=args.long_examples,
                )
                if args.only_dump_examples:
                    continue
                print(json.dumps(stats, indent=4))

                if global_stats is not None:
                    update_stats(global_stats, stats)
        except Exception as err:
            raise RuntimeError(f"Error while iterating on '{subname}'") from err

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
