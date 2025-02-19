import random

import datasets
import regex as re
from tqdm import tqdm

from data import (
    DataIterator,
    DataIteratorConcat,
    get_datasets,
    test_iterator,
)


class BenchmarkDataIterator(DataIteratorConcat):

    """
    A DataIterator for a benchmark dataset from the Hugging Face datasets library.

    It acts as an iterator which yields examples like:
    {
        "prompt": "A man is sitting on a roof. he",
        "positive": ["starts pulling up roofing on a roof."],
        "negative": ["is ripping level tiles off."],
    }

    Parameters:
    - hf_repo_name: str
        The name of the Hugging Face repository containing the dataset
    - hf_repo_kwargs: dict
        The keyword arguments to pass to the datasets.load_dataset function
    - preprocess: function
        A function to preprocess each example in the dataset
    - splits: list of str
        The split(s) to include in the dataset
    """

    def __init__(
        self,
        hf_repo_kargs,
        hf_repo_kwargs=None,
        preprocess=lambda x: x,
        filter_fn=None,
        splits=None,
        name=None,
    ):
        """
        Initialize the BenchmarkDataIterator.

        Args:
            hf_repo_name (str): The name of the Hugging Face repository containing the dataset
            hf_repo_kwargs (dict): The options to pass to the datasets.load_dataset function
            preprocess (function): A function to preprocess each example in the dataset,
                which produces a dictionary with keys "prompt", "positive" (and "negative" optionally)
            filter_fn (function): A function to filter out examples from the dataset
            splits (list of str): The split(s) to include in the dataset (default: 'validation')
        """
        # Process input arguments
        if hf_repo_kwargs is None:
            hf_repo_kwargs = {}
        if isinstance(hf_repo_kargs, str):
            hf_repo_kargs = [hf_repo_kargs]
        assert isinstance(hf_repo_kargs, list) and len(hf_repo_kargs)
        if isinstance(splits, str):
            splits = [splits]
        elif splits is None:
            splits = ["validation"]

        if name is None:
            name = hf_repo_kargs[0].split("/")[-1]

        hf_dataset = datasets.load_dataset(*hf_repo_kargs, **hf_repo_kwargs)
        it_datasets = []
        for split in splits:
            if split in hf_dataset:
                dataset = hf_dataset[split]
                it_datasets.append(
                    DataIterator(
                        dataset,
                        name=(name + "/" + split).replace("/", "--"),
                        preprocess=preprocess,
                        filter_fn=filter_fn,
                        key=None,
                    )
                )
        assert len(it_datasets), f"No data found with parameters {hf_repo_kargs=}, {hf_repo_kwargs=}, {splits=}"
        super().__init__(it_datasets, name=name)


class DataIteratorARC(DataIteratorConcat):
    def __init__(self, splits="validation", levels=None, **kwargs):
        if levels is None:
            levels = ["ARC-Challenge", "ARC-Easy"]
        super().__init__(
            [
                BenchmarkDataIterator(
                    ["allenai/ai2_arc", level],
                    preprocess=preprocess_arc,
                    filter_fn=filter_unlabeled,
                    splits=splits,
                    name=f"ARC--{level}",
                    **kwargs,
                )
                for level in levels
            ],
            name="ARC",
        )


class DataIteratorARCFrBench(BenchmarkDataIterator):
    def __init__(self, splits="validation", **kwargs):
        super().__init__(
            "manu/french_bench_arc_challenge",
            preprocess=preprocess_arc_frbench,
            filter_fn=filter_unlabeled,
            splits=splits,
            **kwargs,
        )


class DataIteratorHellaswag(BenchmarkDataIterator):
    def __init__(self, splits="validation", **kwargs):
        super().__init__(
            "Rowan/hellaswag", preprocess=preprocess_hellaswag, filter_fn=filter_unlabeled, splits=splits, **kwargs
        )


class DataIteratorHellaswagFrBench(BenchmarkDataIterator):
    def __init__(self, splits="validation", **kwargs):
        super().__init__(
            "manu/french_bench_hellaswag",
            preprocess=preprocess_hellaswag,
            filter_fn=filter_unlabeled,
            splits=splits,
            **kwargs,
        )


class DataIteratorMMLU(DataIteratorConcat):
    def __init__(self, splits="validation", subjects=None, **kwargs):
        repo_name = "cais/mmlu"
        if subjects is None:
            # Take all subjects
            config = datasets.load_dataset_builder(repo_name, "all")
            subjects = [c.name for c in config.BUILDER_CONFIGS]
            subjects = [s for s in subjects if s not in ["all"]]
            if "train" not in splits:
                subjects = [s for s in subjects if s not in ["auxiliary_train"]]
        assert len(subjects), "No subjects found"
        split_name = splits if isinstance(splits, str) else "_".join(splits)
        dataset_name = f"MMLU--{split_name}"
        super().__init__(
            [
                BenchmarkDataIterator(
                    [repo_name, subject],
                    preprocess=preprocess_mmlu,
                    filter_fn=filter_unlabeled,
                    splits=splits,
                    name=f"{dataset_name}--{subject}",
                    **kwargs,
                )
                for subject in tqdm(subjects, desc="Initializing MMLU dataset...")
            ],
            name=dataset_name,
        )


class DataIteratorMMMLU(BenchmarkDataIterator):
    def __init__(self, splits="val", language="fr", **kwargs):  # config: "de", "en", "es", "fr", "it"
        super().__init__(
            ["alexandrainst/m_mmlu", language],
            preprocess=preprocess_mmmlu,
            filter_fn=filter_unlabeled,
            splits=splits,
            **kwargs,
        )


class DataIteratorOpenbook(BenchmarkDataIterator):
    def __init__(self, splits="validation", section="additional", **kwargs):  # only "additional" contains "fact1"
        super().__init__(
            ["allenai/openbookqa", section],
            preprocess=preprocess_openbook,
            filter_fn=filter_unlabeled,
            splits=splits,
            **kwargs,
        )


def get_benchmark_datasets(name="all", **kwargs):
    if name in ["all", ["all"]]:
        name = [
            "a_r_c",
            "a_r_c_fr_bench",
            "hellaswag",
            "hellaswag_fr_bench",
            "m_m_l_u",
            "m_m_m_l_u",  # French
            "openbook",
        ]
    return get_datasets(name, scope=globals(), **kwargs)


#########################################
####### Dataset preprocess ##############


def preprocess_arc(data):
    data["answer"] = data.pop("answerKey")
    return preprocess_generic(data)


def preprocess_arc_frbench(data):
    data["answer"] = data.pop("answerKey")
    num_choices = len(data["choices"])
    res_possible = ["A", "B", "C", "D", "E"]
    data["choices"] = {"text": data["choices"], "label": res_possible[:num_choices]}
    return preprocess_generic(data)


def preprocess_hellaswag(data):
    data["question"] = data.pop("ctx")
    data["answer"] = data.pop("label")
    if not data["answer"]:
        return {}
    data["choices"] = {"text": data["endings"], "label": [str(n) for n in range(len(data["endings"]))]}
    return preprocess_generic(data)


def preprocess_mmlu(data):
    choices = list(data["choices"])
    assert len(choices) == 4
    data["choices"] = {
        "text": choices,
        "label": ["A", "B", "C", "D"],
    }
    return preprocess_generic(data)


def preprocess_mmmlu(data):
    prompt = data["instruction"]
    label = data["answer"].lower()
    positive = []
    negative = []
    for letter in ["a", "b", "c", "d"]:
        choice = data[f"option_{letter}"]
        if choice == data[f"option_{label}"]:
            positive.append(choice)
        else:
            negative.append(choice)
    assert len(positive), f"Problem with {data}"
    assert len(negative), f"Problem with {data}"

    return {
        "prompt": remove_annotation_from_text(prompt),
        "positive": remove_annotation_from_text(positive),
        "negative": remove_annotation_from_text(negative),
    }


def preprocess_openbook(data):
    data["question"] = data["fact1"] + " " + data["question_stem"]
    data["answer"] = data.pop("answerKey")
    return preprocess_generic(data)


#########################################


def filter_unlabeled(data):
    return data.get("positive")


def remove_annotation_from_text(text):
    if isinstance(text, list):
        return [remove_annotation_from_text(t) for t in text]

    def dot_or_nothing(match):
        if "." in match.group(1):
            return match.group(1)
        return "."

    text = re.sub(r"(\.?)(\s*)(\[[a-zA-Z]*\])", dot_or_nothing, text).lstrip(" .")
    return text


def preprocess_generic(data, include_labels="random"):
    prompt = data["question"]
    label = data["answer"]
    res_labels = list(data["choices"]["label"])
    res_text = list(data["choices"]["text"])

    assert len(res_text) == len(res_labels)

    if isinstance(label, int) and label not in res_labels:
        idx = label
    else:
        assert label in res_labels, f"problem with {data}"
        idx = res_labels.index(label)

    if include_labels == "random":
        include_labels = random.random() > 0.5

    if include_labels:
        separator = random.choice([" ", "\n"])
        prompt += f". Here are the choices:{separator}" + separator.join(
            [f"{y}: {x}." for x, y in zip(res_text, res_labels)]
        )
        for i, y in enumerate(res_labels):
            res_text[i] = f"{y}: {res_text[i]}."

    assert idx >= 0 and idx < len(res_text), f"Problem with {dict(data)} ({idx=}, {label=}, {len(res_text)=})"
    positive = [res_text[idx]]
    negative = res_text[:idx] + res_text[idx + 1 :]

    return {
        "prompt": remove_annotation_from_text(prompt),
        "positive": remove_annotation_from_text(positive),
        "negative": remove_annotation_from_text(negative),
    }


if __name__ == "__main__":
    import argparse
    import json

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
        "--splits",
        type=str,
        default=None,
        help="Which split to use (ex: 'train', 'validation', 'test', 'validation,test', ...)",
    )
    args = parser.parse_args()

    kwargs = {}
    if args.splits:
        kwargs["splits"] = args.splits.split(",")

    all_datasets = get_benchmark_datasets(args.dataset, **kwargs)
    for dataset in all_datasets:
        stats = test_iterator(
            dataset,
            name=dataset.name,
            # folder=args.folder,
            # ignore_if_exists=args.ignore_if_exists,
            # num_examples=num_examples,
            # only_dump_examples=args.only_dump_examples,
            # prefix_example_files=prefix_example_files,
        )
        print(json.dumps(stats, indent=4))
