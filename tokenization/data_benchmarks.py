import datasets
import regex as re

from data import DataIterator, DataIteratorConcat


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
        hf_repo_name,
        hf_repo_kwargs=None,
        preprocess=lambda x: x,
        filter_fn=None,
        splits=None,
    ):
        """
        Initialize the BenchmarkDataIterator.

        Args:
            hf_repo_name (str): The name of the Hugging Face repository containing the dataset
            hf_repo_kwargs (dict): The options to pass to the datasets.load_dataset function
            preprocess (function): A function to preprocess each example in the dataset,
                which produces a dictionary with keys "prompt", "positive" (and "negative" optionally)
            filter_fn (function): A function to filter out examples from the dataset
            splits (list of str): The split(s) to include in the dataset
        """
        # Process input arguments
        if hf_repo_kwargs is None:
            hf_repo_kwargs = {}
        if isinstance(splits, str):
            splits = [splits]
        elif splits is None:
            splits = ["validation"]

        hf_dataset = datasets.load_dataset(hf_repo_name, **hf_repo_kwargs)
        it_datasets = []
        for split in splits:
            if split in hf_dataset:
                dataset = hf_dataset[split]
                it_datasets.append(
                    DataIterator(
                        dataset,
                        name=(hf_repo_name + "/" + split).replace("/", "--"),
                        preprocess=preprocess,
                        filter_fn=filter_fn,
                        key=None,
                    )
                )
        super().__init__(it_datasets)


def remove_annotation_from_text(text):
    if isinstance(text, list):
        return [remove_annotation_from_text(t) for t in text]

    def dot_or_nothing(match):
        if "." in match.group(1):
            return ""
        return "."

    text = re.sub(r"(\.?)(\s*)(\[[a-zA-Z]*\])", dot_or_nothing, text).lstrip(" .")
    return text


def preprocess_hellaswag(data):
    prompt = data["ctx"]
    label = data["label"]
    endings = data["endings"]
    if not label:
        return {}
    label = int(label)
    positive = [endings[label]]
    negative = endings[:label] + endings[label + 1 :]
    return {
        "prompt": remove_annotation_from_text(prompt),
        "positive": remove_annotation_from_text(positive),
        "negative": remove_annotation_from_text(negative),
    }


def filter_hellaswag(data):
    return data.get("positive")


class DataIteratorHellaswag(BenchmarkDataIterator):
    def __init__(self, splits="validation"):
        super().__init__(
            "Rowan/hellaswag",
            preprocess=preprocess_hellaswag,
            filter_fn=filter_hellaswag,
            splits=splits,
        )


if __name__ == "__main__":
    dataset = DataIteratorHellaswag()

    for data in dataset:
        print(data)
