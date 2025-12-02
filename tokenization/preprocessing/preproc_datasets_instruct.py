import os

import datasets
import regex as re
from preproc_datasets import DataIterator
from preproc_datasets_annealing import INSTRUCT_DATA_PATH
from transformers import AutoTokenizer

tokenizer = None


def tokenizer_from_cache():
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("OpenLLM-France/Lucie-7B-Instruct-v1.1")
    return tokenizer


def convert_to_chat(
    data,
    instruction_col_name="instruction",
    input_col_name="input",
    output_col_name="output",
):
    if (input_col_name is not None) and (data[input_col_name] != ""):
        chat = [
            {"role": "user", "content": f"{data[instruction_col_name]}\n{data[input_col_name]}"},
            {"role": "assistant", "content": f"{data[output_col_name]}"},
        ]
    else:
        chat = [
            {"role": "user", "content": f"{data[instruction_col_name]}"},
            {"role": "assistant", "content": f"{data[output_col_name]}"},
        ]

    text = tokenizer_from_cache().apply_chat_template(chat, tokenize=False)
    data["text"] = text
    return data


def apply_chat_template(data, key):
    text = tokenizer_from_cache().apply_chat_template(data[key], tokenize=False)
    data["text"] = text
    return data


filter_strings = [
    "OpenAI",
    "Open AI",
    "ChatGPT",
    "Chat GPT",
    "GPT-3",
    "GPT3",
    "GPT 3",
    "GPT-4",
    "GPT4",
    "GPT 4",
    "GPT-3.5",
    "GPT3.5",
    "GPT 3.5",
    "BingChat",
    "Bing Chat",
    "LAION",
    "Open Assistant",
    "OpenAssistant",
    "BARD",
    "PaLM",
    "Gemini",
    "Gemma",
    "Google AI",
    "Anthropic",
    "Claude",
    "LLaMA",
    "Meta AI",
    "Mixtral",
    "Mistral",
]


def filter_output_by_keyword(example, key):
    if re.search(r"\b(" + "|".join([s.lower() for s in filter_strings]) + r")\b", example[key].lower()):
        return False
    return True


def filter_conversations_by_keyword(example, key):
    # we filter out conversations that contain some specific strings
    for message in example[key]:
        if message["role"] != "assistant":
            continue
        if re.search(r"\b(" + "|".join([s.lower() for s in filter_strings]) + r")\b", message["content"].lower()):
            return False
    return True


class DataIteratorAlpaca(DataIterator):
    def __init__(self, language="en", streaming=True, **kwargs):
        if language == "en":
            data_path = "yahma/alpaca-cleaned"
        elif language == "fr":
            data_path = "cmh/alpaca_data_cleaned_fr_52k"
        else:
            raise NotImplementedError
        DataIterator.__init__(
            self,
            datasets.load_dataset(
                data_path,
                streaming=streaming,
                split="train",
            ),
            name=f"Alpaca:{language}",
            preprocess=convert_to_chat,
            filter_fn=lambda x: filter_output_by_keyword(x, "output"),
            **kwargs,
        )


class DataIteratorLongAlpaca(DataIterator):
    def __init__(self, language="en", streaming=True, **kwargs):
        DataIterator.__init__(
            self,
            datasets.load_dataset(
                "Yukang/LongAlpaca-12k",
                streaming=streaming,
                split="train",
            ),
            name=f"LongAlpaca:{language}",
            preprocess=convert_to_chat,
            **kwargs,
        )


class DataIteratorAyaDatasetChat(DataIterator):
    def __init__(self, language="en", streaming=True, **kwargs):
        name = f"AyaDatasetChat:{language}"
        language_map = {"en": "English", "fr": "French", "es": "Spanish", "it": "Italian", "de": "German"}
        DataIterator.__init__(
            self,
            datasets.load_dataset(
                "CohereForAI/aya_dataset",
                streaming=streaming,
                split="train",
            ),
            name=name,
            filter_fn=lambda x: x["language"] == language_map[language],
            preprocess=lambda x: convert_to_chat(
                x, instruction_col_name="inputs", input_col_name=None, output_col_name="targets"
            ),
            **kwargs,
        )


class DataIteratorAyaChat(DataIterator):
    def __init__(self, language="en", streaming=True, **kwargs):
        name = f"AyaChat:{language}"
        print(name)
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
            preprocess=lambda x: convert_to_chat(
                x, instruction_col_name="inputs", input_col_name=None, output_col_name="targets"
            ),
            **kwargs,
        )


class DataIteratorDolly(DataIterator):
    def __init__(self, language="en", streaming=True, **kwargs):
        DataIterator.__init__(
            self,
            datasets.load_dataset(
                "argilla/databricks-dolly-15k-curated-multilingual",
                streaming=streaming,
                split=language,
            ),
            name=f"Dolly:{language}",
            preprocess=lambda x: convert_to_chat(x, input_col_name="context", output_col_name="response"),
            **kwargs,
        )


class DataIteratorFlanv2Converted(DataIterator):
    def __init__(self, language="en", streaming=True, **kwargs):
        name = f"Flanv2Converted:{language}"
        with open(
            os.path.join(
                INSTRUCT_DATA_PATH, "flan/task_name_to_exclude_for_languages_ai2-adapt-dev.flan_v2_converted.txt"
            )
        ) as f:
            excluded_tasks = f.read().splitlines()
        excluded_tasks = set(excluded_tasks)
        DataIterator.__init__(
            self,
            datasets.load_dataset(
                "ai2-adapt-dev/flan_v2_converted",
                streaming=streaming,
                split="train",
            ),
            name=name,
            filter_fn=lambda x: x["_task_name"] not in excluded_tasks,
            preprocess=lambda data: apply_chat_template(data, "messages"),
            **kwargs,
        )


class DataIteratorEns(DataIterator):
    def __init__(self, language="fr", streaming=True, **kwargs):
        DataIterator.__init__(
            self,
            datasets.load_dataset(
                "Gael540/dataSet_ens_sup_fr-v1",
                streaming=streaming,
                split="train",
            ),
            name=f"Ens:{language}",
            preprocess=lambda x: convert_to_chat(
                x, instruction_col_name="question", input_col_name=None, output_col_name="answer"
            ),
            **kwargs,
        )


class DataIteratorOasst1(DataIterator):
    def __init__(self, language="en", streaming=True, **kwargs):
        name = f"Oasst1:{language}"
        DataIterator.__init__(
            self,
            datasets.Dataset.from_json(
                os.path.join(INSTRUCT_DATA_PATH, "oasst1/oasst1_format-2.jsonl"),
            ),
            name=name,
            filter_fn=lambda x: x["language"] == language,
            preprocess=lambda data: apply_chat_template(data, "messages"),
            **kwargs,
        )


class DataIteratorFilteredOasst(DataIterator):
    def __init__(self, language="en", streaming=True, **kwargs):
        name = f"FilteredOasst:{language}"
        print("load dataset")
        dataset = datasets.Dataset.from_json(
            os.path.join(INSTRUCT_DATA_PATH, "oasst1/oasst1_format-2.jsonl"),
        )
        print("convert to pandas")
        df = dataset.to_pandas()
        print("deduplicate")
        df = df[df.duplicated("messages")]
        print("convert to dataset")
        dataset = datasets.Dataset.from_pandas(df)
        print(dataset)
        DataIterator.__init__(
            self,
            dataset,
            name=name,
            filter_fn=lambda x: (x["language"] == language) & (filter_conversations_by_keyword(x, "messages")),
            preprocess=lambda data: apply_chat_template(data, "messages"),
            **kwargs,
        )


class DataIteratorPiaf(DataIterator):
    def __init__(self, language="fr", streaming=True, **kwargs):
        name = f"Piaf:{language}"
        DataIterator.__init__(
            self,
            datasets.Dataset.from_json(
                os.path.join(INSTRUCT_DATA_PATH, "piaf/piaf-v1.2_instruct.jsonl"),
            ),
            name=name,
            preprocess=lambda data: apply_chat_template(data, "messages"),
            **kwargs,
        )


class DataIteratorOracle(DataIterator):
    def __init__(self, language="fr", streaming=True, **kwargs):
        name = f"Oracle:{language}"
        DataIterator.__init__(
            self,
            datasets.Dataset.from_json(
                os.path.join(INSTRUCT_DATA_PATH, "oracle/oracle_instruct.jsonl"),
            ),
            name=name,
            preprocess=lambda data: apply_chat_template(data, "messages"),
            **kwargs,
        )


class DataIteratorHardCoded(DataIterator):
    def __init__(self, language="fr", **kwargs):
        name = f"HardCoded:{language}"
        language_map = {"fr": "french", "en": "english"}
        dataset_name = f"hard_coded/openllm_{language_map[language]}.jsonl"
        print(INSTRUCT_DATA_PATH)
        print(dataset_name)
        DataIterator.__init__(
            self,
            datasets.Dataset.from_json(
                os.path.join(INSTRUCT_DATA_PATH, dataset_name),
            ),
            name=name,
            preprocess=lambda data: apply_chat_template(data, "messages"),
            **kwargs,
        )


class DataIteratorCroissantAlignedInstruct(DataIterator):
    def __init__(self, **kwargs):
        name = "CroissantAlignedInstruct"
        DataIterator.__init__(
            self,
            datasets.load_dataset(
                "parquet",
                data_dir=os.path.join(INSTRUCT_DATA_PATH, "traduction/CroissantAligned/v1.2"),
            )["train"],
            name=name,
            preprocess=lambda data: apply_chat_template(data, "messages"),
            **kwargs,
        )


class DataIteratorWildChat(DataIterator):
    def __init__(self, language="fr", streaming=True, **kwargs):
        name = f"WildChat:{language}"
        DataIterator.__init__(
            self,
            datasets.load_dataset(
                "allenai/WildChat-1M",
                streaming=streaming,
                split="train",
            ),
            name=name,
            preprocess=lambda data: apply_chat_template(data, "conversation"),
            filter_fn=lambda x: (x["language"] == "French") & (filter_conversations_by_keyword(x, "conversation")),
            **kwargs,
        )


class DataIteratorTulu3(DataIterator):
    def __init__(self, language="en", streaming=True, **kwargs):
        name = f"Tulu3:{language}"
        DataIterator.__init__(
            self,
            datasets.load_dataset(
                "allenai/tulu-3-sft-mixture",
                streaming=streaming,
                split="train",
            ),
            name=name,
            preprocess=lambda data: apply_chat_template(data, "messages"),
            filter_fn=lambda x: x["source"]
            not in ["ai2-adapt-dev/tulu_v3.9_aya_100k", "ai2-adapt-dev/oasst1_converted", ""],
            **kwargs,
        )


class DataIteratorPersonasMath(DataIterator):
    def __init__(self, language="en", streaming=True, **kwargs):
        name = f"PersonasMath:{language}"
        DataIterator.__init__(
            self,
            datasets.load_dataset(
                "allenai/tulu-3-sft-personas-math",
                streaming=streaming,
                split="train",
            ),
            name=name,
            preprocess=lambda data: apply_chat_template(data, "messages"),
            filter_fn=lambda x: filter_conversations_by_keyword(x, "messages"),
            **kwargs,
        )


class DataIteratorPersonasMathGrade(DataIterator):
    def __init__(self, language="en", streaming=True, **kwargs):
        name = f"PersonasMathGrade:{language}"
        DataIterator.__init__(
            self,
            datasets.load_dataset(
                "allenai/tulu-3-sft-personas-math-grade",
                streaming=streaming,
                split="train",
            ),
            name=name,
            preprocess=lambda data: apply_chat_template(data, "messages"),
            filter_fn=lambda x: filter_conversations_by_keyword(x, "messages"),
            **kwargs,
        )


class DataIteratorOpenHermes(DataIterator):
    def __init__(self, language="en", **kwargs):
        name = f"OpenHermes:{language}"
        DataIterator.__init__(
            self,
            datasets.load_dataset(
                "teknium/OpenHermes-2.5",
                split="train",
            ),
            name=name,
            preprocess=lambda data: apply_chat_template(
                DataIteratorOpenHermes.rename_conversations(data), "conversations"
            ),
            filter_fn=lambda x: filter_conversations_by_keyword(x, "conversations"),
            **kwargs,
        )

    @staticmethod
    def rename_conversations(data):
        data["conversations"] = [
            {"role": "user" if item["from"] == "human" else "assistant", "content": item["value"]}
            for item in data["conversations"]
        ]
        return data


def preproc_magpie(data):
    messages = []
    for conv in data["conversations"]:
        messages.append(
            {
                "role": "user" if conv["from"] == "human" else "assistant",
                "content": conv["value"],
            }
        )
    data["messages"] = messages
    data = apply_chat_template(data, "messages")
    return data


class DataIteratorMagpieGemma(DataIterator):
    def __init__(self, language="en", streaming=True, **kwargs):
        name = f"MagpieGemma:{language}"
        DataIterator.__init__(
            self,
            datasets.load_dataset(
                "Magpie-Align/Magpie-Gemma2-Pro-200K-Filtered",
                streaming=streaming,
                split="train",
            ),
            name=name,
            preprocess=preproc_magpie,
            filter_fn=lambda x: (x["language"] == "EN") & (filter_conversations_by_keyword(x, "messages")),
            **kwargs,
        )
