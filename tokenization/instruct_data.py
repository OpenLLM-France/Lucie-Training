import os
import random
import re

import pandas as pd

from data import (
    DataIterator,
    DataIteratorConcat,
    # Overloaded classes
    DataIteratorCroissantAligned,
    _asset_folder,
    decompose_datasets,
    # needed for the main to write parquets
    get_datasets,
    set_data_iterator_prefix,
)


class InstructDataIterator(DataIterator):
    def __init__(self, dataset, data_to_question_and_answer=None, *kargs, **kwargs):
        if isinstance(dataset, DataIterator):
            # inherit from the dataset
            self.__dict__ = dataset.__dict__
            self.name += "_instruct"
        else:
            # Wrapped "HuggingFace" dataset
            DataIterator.__init__(self, dataset, *kargs, **kwargs)

        # Will process the data just before the end
        self.key = self.data_to_chat_instructions

        # That can be passed externally
        self.data_to_question_and_answer = None
        if data_to_question_and_answer:
            self.data_to_question_and_answer = data_to_question_and_answer

    def data_to_chat_instructions(self, data):
        try:
            question, answer = self.data_to_question_and_answer(data)
            assert isinstance(question, str)
            question_answers = [(question, answer)]
        except (TypeError, AssertionError):
            question_answers = self.data_to_question_and_answer(data)

        turns = []
        assert isinstance(question_answers, list)
        for question, answer in question_answers:
            assert isinstance(question, str)
            assert isinstance(answer, str)
            turns += [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        return turns

    def data_to_question_and_answer(self, data):
        if self.data_to_question_and_answer:
            return self.data_to_question_and_answer(data)
        raise NotImplementedError


class InstructDataIteratorConcat(DataIteratorConcat):
    def __init__(self, datasets, *kargs, **kwargs):
        DataIteratorConcat.__init__(self, datasets, *kargs, **kwargs)
        self.datasets = [
            InstructDataIterator(
                dataset, *kargs, data_to_question_and_answer=self.data_to_question_and_answer, **kwargs
            )
            for dataset in datasets
        ]

    def data_to_question_and_answer(self, data):
        raise NotImplementedError


### Implementation of specific datasets

_asset_croissant_aligned_by_languages = {
    lan: os.path.join(_asset_folder, "instruct", f"translation_{lan}.txt") for lan in ["fr", "en"]
}


class InstructDataIteratorTranslationCroissant(DataIteratorCroissantAligned, InstructDataIteratorConcat):
    def __init__(self, *kargs, **kwargs):
        DataIteratorCroissantAligned.__init__(self, *kargs, **kwargs)

        # Make that an instruction dataset (composite dataset)
        InstructDataIteratorConcat.__init__(self, self.datasets, *kargs, name=self.name + "_instruct", **kwargs)

        # Load resources
        self.instruct_templates = {}
        for lan, filename in _asset_croissant_aligned_by_languages.items():
            assert os.path.exists(filename), f"File {filename} does not exist"
            with open(filename) as f:
                self.instruct_templates[lan] = [line.strip() for line in f.readlines()]

        self.language_strings = {
            "fr": {
                "fr": "français",
                "en": "anglais",
            },
            "en": {
                "fr": "French",
                "en": "English",
            },
        }

    def data_to_question_and_answer(self, data):
        assert "text_fr" in data
        assert "text_en" in data
        text_fr = data["text_fr"]
        text_en = data["text_en"]

        french_first = random.random() < 0.5
        language_from = "fr" if french_first else "en"
        language_to = "en" if french_first else "fr"
        text_from = text_fr if french_first else text_en
        text_to = text_en if french_first else text_fr

        instruction_in_the_same_language = random.random() < 0.5
        language_instruction = language_from if instruction_in_the_same_language else language_to
        set_of_instructions = self.instruct_templates[language_instruction]

        do_capitalize_language = language_instruction == "en"
        if random.random() < 0.1:  # Wrong language case
            do_capitalize_language = not do_capitalize_language

        language_from_str = self.language_strings[language_instruction][language_from]
        language_to_str = self.language_strings[language_instruction][language_to]
        if do_capitalize_language:
            language_from_str = language_from_str.capitalize()
            language_to_str = language_to_str.capitalize()

        instruction = random.choice(set_of_instructions)
        instruction = re.sub("<language_from>", language_from_str, instruction)
        instruction = re.sub("<language_to>", language_to_str, instruction)
        if instruction.endswith("."):
            instruction = instruction[:-1] + random.choice([".", ":", ""])

        separators = random.choice(
            [
                ("\n", ""),
                ("\n\n", ""),
                (' "', '"'),
                (" '", "'"),
                ('\n"', '"'),
                ('\n\n"', '"'),
            ]
        )
        instruction += separators[0] + text_from + separators[1]

        return instruction, text_to


def main_dump_parquet():
    set_data_iterator_prefix("InstructDataIterator", scope=globals())

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", type=str, default="translation")
    parser.add_argument("--high_quality", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    all_datas = get_datasets(args.datasets, high_quality=args.high_quality)
    all_datas = dict(
        dataset_to_key_value(dataset)
        for dataset in decompose_datasets(all_datas, parquet_level=True, return_json_file_if_possible=True)
    )

    for k, data_iterator in all_datas.items():
        random.seed(1969)
        output_filename = k + ".parquet"
        print(f"Generating {output_filename}")
        datas = []
        for i, messages in enumerate(data_iterator):
            if args.verbose and i < 1000:
                for message in messages:
                    print(f"{message['role']}: {cut_long_string(message['content'])}")
            datas.append(messages)
            if len(datas) > 1000:
                break
        pd.DataFrame({"messages": datas}).to_parquet(output_filename)


def cut_long_string(text, max_len=100):
    text = text.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")
    if len(text) > max_len:
        text = text[:max_len] + "..."
    return text


def dataset_to_key_value(dataset):
    if isinstance(dataset, tuple) and len(dataset) == 2:
        return dataset
    else:
        return (dataset.name.replace(":", "--"), dataset)


if __name__ == "__main__":
    main_dump_parquet()

    # set_data_iterator_prefix("InstructDataIterator", scope=globals())
    # main()
