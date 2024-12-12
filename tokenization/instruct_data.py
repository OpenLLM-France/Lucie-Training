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
    def __init__(self, *kargs, verbose=False, debug=False, **kwargs):
        DataIteratorCroissantAligned.__init__(self, *kargs, **kwargs)

        # Make that an instruction dataset (composite dataset)
        InstructDataIteratorConcat.__init__(self, self.datasets, *kargs, name=self.name + "_instruct", **kwargs)

        # Some parameters
        # ----------------
        # - proba that the language of the text to translate is the same as the instruction language
        self.proba_same_language = 0.5
        # - proba that the text to translate is in French (otherwise in English)
        self.proba_french_to_english = 0.5
        # - Probabilities of common mistakes
        self.proba_wrong_case_for_language = (
            0.3  # confusion "english" <-> "English" (right) | "anglais" (right) <-> "Anglais"
        )
        self.proba_wrong_question_mark_fr = 0.5  # don't use space before question mark in French
        # - Probabilities of whether from/to languages are mentioned in the instruction,
        #   depending on whether the instruction is in the **same language as the input text to translate**
        self.proba_by_languages_mentioned_if_different_from_instruct = {
            "no_lang": 4.0 / 7,
            "to": 2.0 / 7,
            "from_and_to": 1.0 / 7,
        }
        # In this second case, we have to specify the target language when it's not the same as the original language
        self.proba_by_languages_mentioned_if_same_as_instruct = {
            "to": 2.0 / 3,
            "from_and_to": 1.0 / 3,
        }

        # Languages strings
        self.language_strings = {
            "fr": {
                "fr": ("français", "francais"),
                "en": "anglais",
            },
            "en": {
                "fr": "French",
                "en": "English",
            },
        }

        # Tricks to remove tags from input instructions (to make more instructions)
        patterns_to_remove_from_tag = {
            # FRENCH
            "fr": [(", qui est en <language_from>,", "")]
            + [(rf"(, | )?{determinant} <language_from>,?", "") for determinant in ["de", "du", "en"]]
            + [(rf"\b({word}) <language_from>", r"\1") for word in ["texte"]],
            # ENGLISH
            "en": [(", which is in <language_from>,", "")]
            + [(rf"(, | )?{determinant} <language_from>,?", "") for determinant in ["from", "in"]]
            + [(rf"<language_from> ({word})\b", r"\1") for word in ["text"]],
        }
        patterns_to_remove_to_tag = {
            # FRENCH
            "fr": [(r"(, | )?(en|vers de|vers du) <language_to>,?", "")],
            # ENGLISH
            "en": [(r"(, | )?(in)?to <language_to>,?", "")],
        }

        self.regex_to_remove_from_tag = {
            lan: [(re.compile(pattern, re.IGNORECASE), to) for (pattern, to) in patterns]
            for lan, patterns in patterns_to_remove_from_tag.items()
        }
        self.regex_to_remove_to_tag = {
            lan: [(re.compile(pattern, re.IGNORECASE), to) for (pattern, to) in patterns]
            for lan, patterns in patterns_to_remove_to_tag.items()
        }

        def remove_tag(text, list_of_regex):
            if isinstance(text, list):
                return [remove_tag(t, list_of_regex) for t in text]
            new_text = None
            for regex, to in list_of_regex:
                if regex.search(text):
                    new_text = regex.sub(to, text)
                    break
            return new_text if new_text != text else None

        # Load instruction templates, and sort them by whether they have <language_from> and/or <language_to>
        self.instruct_templates = {
            lan: {k: [] for k in ["no_lang", "to", "from_and_to"]} for lan in self.language_strings
        }

        for lan, filename in _asset_croissant_aligned_by_languages.items():
            assert os.path.exists(filename), f"File {filename} does not exist"
            with open(filename) as f:
                for template in f.readlines():
                    template = template.strip()
                    if not template:
                        continue
                    has_from = "<language_from>" in template
                    has_to = "<language_to>" in template
                    if has_from and has_to:
                        self.instruct_templates[lan]["from_and_to"].append(template)
                        template_without_from = remove_tag(template, self.regex_to_remove_from_tag[lan])
                        if template_without_from:
                            self.instruct_templates[lan]["to"].append(template_without_from)
                            template_without_from_and_to = remove_tag(
                                template_without_from, self.regex_to_remove_to_tag[lan]
                            )
                            if template_without_from_and_to:
                                self.instruct_templates[lan]["no_lang"].append(template_without_from_and_to)
                    elif has_to:
                        self.instruct_templates[lan]["to"].append(template)
                    elif has_from:
                        raise ValueError(f"Template {template} has <language_from> but not <language_to>")
                    else:
                        self.instruct_templates[lan]["no_lang"].append(template)
            # Check it is not empty and remove duplicates
            for k in self.instruct_templates[lan]:
                assert self.instruct_templates[lan][k], f"No instruction template found for {k}"
                self.instruct_templates[lan][k] = sorted(set(self.instruct_templates[lan][k]))

        # Set probabilities inversely proportional to the square of the number of characters
        def compute_weights(instructions):
            lengths = [len(instruction) for instruction in instructions]
            assert min(lengths) > 0
            weights = [1 / numchar**2 for numchar in lengths]
            total = sum(weights)
            assert total > 0
            return [w / total for w in weights]

        self.intruct_template_probas = {
            lan: {k: compute_weights(instructions) for k, instructions in self.instruct_templates[lan].items()}
            for lan in self.instruct_templates
        }

        self.verbose = verbose

        if debug:
            for lan in self.instruct_templates:
                for type in self.instruct_templates[lan]:
                    instructions = self.instruct_templates[lan][type]
                    weights = self.intruct_template_probas[lan][type]
                    assert len(instructions) == len(weights)
                    max_weight = max(weights)
                    with open(f"tmp_debug_instructions_{type}_{lan}.txt", "w") as f:
                        for w, instruction in zip(weights, instructions):
                            print(f"{w * 100 / max_weight:.2f} {instruction}", file=f)

        if verbose:
            for lan in self.instruct_templates:
                for k in self.instruct_templates[lan]:
                    k_str = {
                        "no_lang": "without language",
                        "to": "with language to",
                        "from_and_to": "with language from and to",
                    }[k]
                    print(f"Loaded {len(self.instruct_templates[lan][k])} templates {k_str} for {lan}")

    def data_to_question_and_answer(self, data):
        assert "text_fr" in data
        assert "text_en" in data
        text_fr = data["text_fr"]
        text_en = data["text_en"]

        french_first = random.random() < self.proba_french_to_english
        language_input = "fr" if french_first else "en"
        language_output = "en" if french_first else "fr"
        text_input = text_fr if french_first else text_en
        text_translated = text_en if french_first else text_fr

        instruction_has_input_language = random.random() < self.proba_same_language
        language_instruction = language_input if instruction_has_input_language else language_output

        # Choose whether to mention input/output languages in the instruction or not
        proba_by_languages_mentioned = (
            self.proba_by_languages_mentioned_if_same_as_instruct
            if instruction_has_input_language
            else self.proba_by_languages_mentioned_if_different_from_instruct
        )
        type = random.choices(
            list(proba_by_languages_mentioned.keys()), weights=list(proba_by_languages_mentioned.values())
        )[0]

        # Choose the instruction template among available ones
        instruction = random.choices(
            self.instruct_templates[language_instruction][type],
            weights=self.intruct_template_probas[language_instruction][type],
        )[0]

        # Get the language names, and noise them a bit
        language_input_str = self.language_strings[language_instruction][language_input]
        language_output_str = self.language_strings[language_instruction][language_output]
        # - Sometimes there are several options (["français",  "francais"] -> random choice)
        if not isinstance(language_input_str, str):
            language_input_str = random.choice(language_input_str)
        if not isinstance(language_output_str, str):
            language_output_str = random.choice(language_output_str)
        # - Capitalize (or not) the language
        should_capitalize = language_instruction == "en"
        if random.random() < self.proba_wrong_case_for_language:
            do_capitalize = not should_capitalize
            language_input_str = language_input_str.capitalize() if do_capitalize else language_input_str.lower()
            language_output_str = language_output_str.capitalize() if do_capitalize else language_output_str.lower()

        # Make the substitutions
        # - First language-specific fixes
        if language_instruction == "fr":
            # Contractions needed (de -> d', du -> de l')
            if language_input_str[0] in "aeiouyAEIOUY":  # only when the word starts with a vowel
                instruction = re.sub(r"\b([Dd])e <language_from>", r"\1'" + language_input_str, instruction)
                instruction = re.sub(r"\b([Dd])u <language_from>", r"\1e l'" + language_input_str, instruction)
            # Sometimes remove space before question mark in French
            if instruction.endswith("?") and random.random() < self.proba_wrong_question_mark_fr:
                instruction = instruction[:-1].rstrip() + "?"
            # Contractions needed (au -> à l', du -> de l')
            if language_output_str[0] in "aeiouyAEIOUY":  # only when the word starts with a vowel
                instruction = re.sub(r"\bau <language_to>", r"à l'" + language_output_str, instruction)
                instruction = re.sub(r"\bdu <language_to>", r"de l'" + language_output_str, instruction)
        # - Then the general substitutions
        instruction = re.sub("<language_from>", language_input_str, instruction)
        instruction = re.sub("<language_to>", language_output_str, instruction)

        # Additional data augmentation
        # - Choose random sentence ending
        if instruction.endswith("."):
            instruction = instruction[:-1] + random.choices([".", ":", ""], weights=[0.25, 0.35, 0.4])[0]
        elif instruction.endswith("?"):
            instruction = instruction[:-1] + random.choices(["?", ""], weights=[0.7, 0.3])[0]
        # - Choose random separation between the instruction and the text to translate
        separators = random.choice(
            [
                ("\n", ""),
                ("\n\n", ""),
                (' "', '"'),
                (" «", "»"),
                (" '", "'"),
                ('\n"', '"'),
                ('\n\n"', '"'),
                ("\n\n«", "»"),
            ]
        )

        # Make the final instruction and response
        instruction += separators[0] + text_input + separators[1]

        output_prefix = separators[0].strip()
        output_suffix = separators[1].strip()
        response = output_prefix + text_translated + output_suffix

        if self.verbose:
            short_text_input = f"{text_input[:10]:10}[[... ({language_input})]]"
            short_text_translated = f"{text_translated[:10]:10}[[... ({language_output})]]"
            debug_print = (
                (instruction + "[[]]" + response)
                .replace(text_input, short_text_input)
                .replace(text_translated, short_text_translated)
                .replace("\n", "\\n")
                .replace("\t", "\\t")
                .replace("[[]]", "\n -> ")
            )
            print(debug_print)
            # if "\n" in debug_print:
            #     import pdb; pdb.set_trace()

        return instruction, response


def main_dump_parquet():
    set_data_iterator_prefix("InstructDataIterator", scope=globals())

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", type=str, default="translation")
    parser.add_argument("--high_quality", action="store_true")
    parser.add_argument("--max_per_parquet", default=None, type=int, help="Maximum number of examples per parquet file")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    all_datas = get_datasets(args.datasets, high_quality=args.high_quality, debug=args.debug, verbose=args.debug)
    all_datas = dict(
        dataset_to_key_value(dataset)
        for dataset in decompose_datasets(all_datas, parquet_level=True, return_json_file_if_possible=True)
    )

    if args.debug and not args.max_per_parquet:
        args.max_per_parquet = 1000

    random.seed(1969)
    for k, data_iterator in all_datas.items():
        output_filename = k + ".parquet"
        print(f"Generating {output_filename}")
        datas = []
        for i, messages in enumerate(data_iterator):
            if args.max_per_parquet and i >= args.max_per_parquet:
                break
            datas.append(messages)
        pd.DataFrame({"messages": datas}).to_parquet(output_filename)
        if args.debug:
            break


def dataset_to_key_value(dataset):
    if isinstance(dataset, tuple) and len(dataset) == 2:
        return dataset
    else:
        return (dataset.name.replace(":", "--"), dataset)


if __name__ == "__main__":
    main_dump_parquet()

    # set_data_iterator_prefix("InstructDataIterator", scope=globals())
    # main()
