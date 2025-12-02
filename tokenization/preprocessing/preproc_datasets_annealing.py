import os

import datasets
from preproc_datasets import DataIterator, DataIteratorConcat

INSTRUCT_DATA_PATH = os.environ.get("INSTRUCT_DATA_PATH")
if not INSTRUCT_DATA_PATH:
    for path in [
        "/lustre/fsn1/projects/rech/qgz/commun/instruct",  # JZ
        "/data-server/datasets/text/raw/multilang/Lucie/instruct",  # BB
    ]:
        if os.path.isdir(path):
            INSTRUCT_DATA_PATH = path
            break
    if not INSTRUCT_DATA_PATH:
        raise RuntimeError("No instruct data path found. You can set it using INSTRUCT_DATA_PATH environment variable.")


class DataIteratorArxiver(DataIterator):
    def __init__(self, language="en", streaming=True, **kwargs):
        name = f"arxiver:{language}"
        DataIterator.__init__(
            self,
            datasets.load_dataset(
                "neuralwork/arxiver",
                streaming=streaming,
                split="train",
            ),
            name=name,
            key="markdown",
            **kwargs,
        )


class DataIteratorOpenWebMath(DataIterator):
    def __init__(self, language="en", streaming=True, **kwargs):
        name = f"OpenWebMath:{language}"
        DataIterator.__init__(
            self,
            datasets.load_dataset(
                "open-web-math/open-web-math",
                streaming=streaming,
                split="train",
            ),
            name=name,
            **kwargs,
        )


class DataIteratorStackMathQA(DataIterator):
    def __init__(self, language="en", streaming=True, **kwargs):
        name = f"StackMathQA:{language}"
        DataIterator.__init__(
            self,
            datasets.load_dataset(
                "math-ai/StackMathQA",
                streaming=streaming,
                split="train",
            ),
            name=name,
            preprocess=lambda x: {"text": f"### Question:\n{x['Q']}\n### Answer:\n{x['A']}"},
            **kwargs,
        )


class DataIteratorAya(DataIterator):
    def __init__(self, language="en", streaming=True, **kwargs):
        name = f"Aya:{language}"
        language_map = {"en": "english", "fr": "french", "es": "spanish", "it": "italian", "de": "german"}
        DataIterator.__init__(
            self,
            datasets.load_dataset(
                "CohereForAI/aya_collection_language_split",
                language_map[language],
                streaming=streaming,
                split="train",
            ),
            name=name,
            preprocess=lambda x: {"text": x["inputs"] + " " + x["targets"]},
            **kwargs,
        )


class DataIteratorFlanv2(DataIteratorConcat):
    def __init__(self, language="en", streaming=True, **kwargs):
        sampling_sizes = {
            "cot_fsopt_data": 20000 * 50,
            "cot_zsopt_data": 20000 * 50,
            "niv2_fsopt_data": 20000 * 50,
            "niv2_zsopt_data": 20000 * 50,
            "flan_fsopt_data": 2000 * 50,
            "flan_zsopt_data": 2000 * 50,
            "t0_fsopt_data": 6000 * 50,
        }

        with open(
            os.path.join(
                INSTRUCT_DATA_PATH, "flan/task_name_to_exclude_for_languages_ai2-adapt-dev.flan_v2_converted.txt"
            )
        ) as f:
            excluded_tasks = f.read().splitlines()
        excluded_tasks = set(excluded_tasks)

        DataIteratorConcat.__init__(
            self,
            [
                DataIterator(
                    datasets.load_dataset(
                        "Open-Orca/FLAN",
                        data_files=f"{subset}/*",
                        streaming=streaming,
                        split="train",
                    ).shuffle(seed=42),
                    name=f"Flanv2:{language}:{subset}",
                    preprocess=lambda x: {"text": x["inputs"] + x["targets"], "_task_name": x["_task_name"]},
                    max_docs=sampling_size,
                    **kwargs,
                )
                for subset, sampling_size in sampling_sizes.items()
            ],
            name=f"Flanv2:{language}",
        )
