import json
import os
import random
import re

import pandas as pd

from data import (
    DataIterator,
    _asset_folder,
    decompose_config,
    norm_config_name,
)


class InstructDataIterator(DataIterator):
    def __init__(self, dataset, data_to_chat_instructions=None, *kargs, **kwargs):
        if isinstance(dataset, DataIterator):
            # Inherit from the dataset
            self.__dict__ = dataset.__dict__
        else:
            # Wrapped "HuggingFace" dataset
            DataIterator.__init__(self, dataset, *kargs, **kwargs)

        # Will process the data just before the end
        self.key = self.data_to_chat_instructions

        # That can be passed externally
        self.data_to_chat_instructions = None
        if data_to_chat_instructions:
            self.data_to_chat_instructions = data_to_chat_instructions

    def data_to_chat_instructions(self, data):
        if self.data_to_chat_instructions:
            return self.data_to_chat_instructions(data)
        raise NotImplementedError


### Implementation of specific datasets

_language_dict = {
    "fr": {
        "fr": "français",
        "en": "anglais",
        "es": "espagnol",
        "de": "allemand",
        "it": "italien",
    },
    "en": {
        "fr": "French",
        "en": "English",
        "es": "Spanish",
        "de": "German",
        "it": "Italian",
    },
    "es": {
        "fr": "francés",
        "en": "inglés",
        "es": "español",
        "de": "alemán",
        "it": "italiano",
    },
    "de": {
        "fr": "Französisch",
        "en": "Englisch",
        "es": "Spanisch",
        "de": "Deutsch",
        "it": "Italienisch",
    },
    "it": {
        "fr": "francese",
        "en": "inglese",
        "es": "spagnolo",
        "de": "tedesco",
        "it": "italiano",
    },
}

_asset_instruct_translation = {
    lan: os.path.join(_asset_folder, "instruct", f"translation_{lan}.txt") for lan in _language_dict
}
_asset_instruct_translation_cont = {
    lan: os.path.join(_asset_folder, "instruct", f"translation_{lan}_cont.txt") for lan in _language_dict
}


class InstructDataIteratorTranslation(InstructDataIterator):
    def __init__(self, *kargs, verbose=False, **kwargs):
        self.dataset = DataIterator(*kargs, **kwargs)

        # Make that an instruction dataset (composite dataset)
        InstructDataIterator.__init__(self, self.dataset, *kargs, name=self.dataset.name + "_instruct", **kwargs)

        # Tricks to remove tags from input instructions (to make more instructions)
        patterns_to_remove_from_tag = {
            # FRENCH
            "fr": [(", qui est en <language_from>,", "")]
            + [(rf"(, | )?{determinant} <language_from>,?", "") for determinant in ["de", "du", "en"]]
            + [(rf"\b({word}) <language_from>", r"\1") for word in ["texte"]],
            # ENGLISH
            "en": [(", which is in <language_from>,", "")]
            + [(rf"(, | )?{determinant} <language_from>,?", "") for determinant in ["from", "in"]]
            + [("<language_from> ({word})\b", r"\1") for word in ["text"]],
            # SPANISH
            "es": [(", escrito en <language_from>,", "")]
            + [(rf"(, | )?{determinant} <language_from>,?", "") for determinant in ["de"]],
            # GERMAN
            "de": [(", der in <language_from> geschrieben ist,", "")]
            + [(rf"(, | )?{determinant} <language_from>,?", "") for determinant in ["von"]],
            # ITALIAN
            "it": [(", scritto in <language_from>,", "")]
            + [(rf"(, | )?{determinant} <language_from>,?", "") for determinant in ["da"]],
        }
        patterns_to_remove_to_tag = {
            # FRENCH
            "fr": [(r"(, | )?(en|vers de|vers du) <language_to>,?", "")],
            # ENGLISH
            "en": [(r"(, | )?(in)?to <language_to>,?", "")],
            # SPANISH
            "es": [(r"(, | )?a <language_to>,?", "")],
            # GERMAN
            "de": [(r"(, | )?nach <language_to>,?", "")],
            # ITALIAN
            "it": [(r"(, | )?in <language_to>,?", "")],
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
                    new_text = regex.sub(to, text).strip()
                    break
            return new_text if new_text != text else None

        # Load instruction templates, and sort them by whether they have <language_from> and/or <language_to>
        self.instruct_templates = {
            lan: {k: [] for k in ["no_lang", "no_lang_with_context", "to", "to_with_context", "from_and_to"]}
            for lan in _language_dict
        }

        for lan, filename in _asset_instruct_translation.items():
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

        for lan, filename in _asset_instruct_translation_cont.items():
            assert os.path.exists(filename), f"File {filename} does not exist"
            with open(filename) as f:
                lines = f.readlines()
                lines = [line.strip() for line in lines]
                lines = [line for line in lines if line]
                self.instruct_templates[lan]["to_with_context"] = lines
                templates_no_lang = [remove_tag(line, self.regex_to_remove_to_tag[lan]) for line in lines]
                templates_no_lang = [line for line in templates_no_lang if line]
                self.instruct_templates[lan]["no_lang_with_context"] = templates_no_lang

        # Check that each set of instruction is not empty and remove duplicates
        for lan in self.instruct_templates:
            found = []
            for k in list(self.instruct_templates[lan].keys()):
                if not self.instruct_templates[lan][k]:
                    print(f"WARNING: No instruction template found for {lan=} / {k}")
                    self.instruct_templates[lan].pop(k)
                    continue
                found.append(k)
                self.instruct_templates[lan][k] = sorted(set(self.instruct_templates[lan][k]))
            if not found:
                raise RuntimeError(f"No template found for {lan=}")

        # Set probabilities inversely proportional to the square of the number of characters
        def compute_weights(instructions):
            if not len(instructions):
                return []
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

        if verbose:
            for lan in self.instruct_templates:
                for k in self.instruct_templates[lan]:
                    k_str = {
                        "no_lang": "without language",
                        "no_lang_with_context": "without language (and a chat history)",
                        "to": "with language to",
                        "to_with_context": "with language to (and a chat history)",
                        "from_and_to": "with language from and to",
                    }[k]
                    print(f"Loaded {len(self.instruct_templates[lan][k])} templates {k_str} for {lan}")

        # Some parameters
        # ----------------
        # - proba that the language of the text to translate is the same as the instruction language
        self.proba_same_language = 0.5
        # - proba that the text to translate is in French (otherwise in English)
        self.proba_french_to_english = 0.5
        # - proba of the instruction coming after the text to translate
        self.proba_instruction_after_text = 0.6
        self.proba_text_to_translate_from_previous_instruct = 0.5
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

        random.seed(1234)

    def data_to_chat_instructions(self, data):
        #########################################
        ### Randomly choose some parameters

        # - choose the input/output languages
        bool_lan_order = random.random() < self.proba_french_to_english
        # - whether the instruction (to translate) comes after the text to translate
        bool_put_instruction_after_text = random.random() < self.proba_instruction_after_text
        # - whether the text to translate comes from the chat history
        bool_translate_from_chat_history = (
            bool_put_instruction_after_text and random.random() < self.proba_text_to_translate_from_previous_instruct
        )
        # - whether the user speaks the same language as the text to translate
        bool_same_input_and_instruct_language = random.random() < self.proba_same_language
        # - whether to mistake about French/french and français/Français
        bool_wrong_case_for_language = random.random() < self.proba_wrong_case_for_language
        # - opening and ending quotes
        opening_quote, ending_quote = random.choice(
            [
                ("", ""),
                ("«", "»"),
                ('"', '"'),
                ("'", "'"),
                ("``", "''"),
                ("“", "”"),
                ("‘", "’"),
            ]
        )
        # - how to separate the instruction from the text to translate
        separator = random.choice(["\n", "\t", "\n\n", ":"] + ([" "] if opening_quote else []))
        if not separator.strip():
            separator *= random.choices([1, 2, 3], weights=[0.85, 0.1, 0.05])[0]
        # - instruction ending punctuation mark
        ending_point = random.choices(
            [".", ":" if not bool_put_instruction_after_text else "", ""], weights=[0.3, 0.2, 0.5]
        )[0]
        ending_question_mark = random.choices(["?", ""], weights=[0.7, 0.3])[0]
        # - choose whether to mention input/output languages in the instruction or not
        proba_by_languages_mentioned = (
            self.proba_by_languages_mentioned_if_same_as_instruct
            if bool_same_input_and_instruct_language
            else self.proba_by_languages_mentioned_if_different_from_instruct
        ).copy()
        # - sometimes use a specific instruction type when there is context before
        if bool_translate_from_chat_history and random.random() < 0.5:
            proba_by_languages_mentioned["to_with_context"] = proba_by_languages_mentioned.pop("to")
            if "no_lang" in proba_by_languages_mentioned:
                proba_by_languages_mentioned["no_lang_with_context"] = proba_by_languages_mentioned.pop("no_lang")
        # - sometimes use lower / upper case only
        case_instruct = random.choices(["lower", "upper", "normal"], weights=[0.18, 0.02, 0.8])[0]
        case_text = random.choices([case_instruct, "normal"], weights=[0.2, 0.8])[0]

        #########################################
        ### Go !

        assert "extra" in data, f"Data does not have 'extra' field: {data}"
        data = json.loads(data["extra"])

        # - load the fields
        if "text_1" in data:
            for k in (
                "text_2",
                "lan_1",
                "lan_2",
            ):
                assert k in data, f"Data does not have '{k}' field: {data}"
            text_1 = data["text_1"]
            text_2 = data["text_2"]
            lan_1 = data["lan_1"]
            lan_2 = data["lan_2"]

        elif "text_fr" in data:
            assert "text_en" in data, f"Data does not have 'text_en' field: {data}"
            text_1 = data["text_fr"]
            text_2 = data["text_en"]
            lan_1 = "fr"
            lan_2 = "en"

        else:
            raise RuntimeError(f"Unexpected data {data}")

        if lan_1 == "en":
            text_1 = normalize_text(text_1, "en")
        if lan_2 == "en":
            text_2 = normalize_text(text_2, "en")

        text_input = text_1 if bool_lan_order else text_2
        text_translated = text_2 if bool_lan_order else text_1

        language_input = lan_1 if bool_lan_order else lan_2
        language_output = lan_2 if bool_lan_order else lan_1
        language_instruction = language_input if bool_same_input_and_instruct_language else language_output

        type = None
        while type not in self.instruct_templates[language_instruction]:
            type = random.choices(
                list(proba_by_languages_mentioned.keys()), weights=list(proba_by_languages_mentioned.values())
            )[0]

        # - choose the instruction template among available ones
        instruction = random.choices(
            self.instruct_templates[language_instruction][type],
            weights=self.intruct_template_probas[language_instruction][type],
        )[0]

        # Get the language names, and noise them a bit
        language_input_str = _language_dict[language_instruction][language_input]
        language_output_str = _language_dict[language_instruction][language_output]
        # - Sometimes there are several options (["français",  "francais"] -> random choice)
        if not isinstance(language_input_str, str):
            language_input_str = random.choice(language_input_str)
        if not isinstance(language_output_str, str):
            language_output_str = random.choice(language_output_str)
        # - Capitalize (or not) the language
        should_capitalize = _language_dict[language_instruction]["fr"][0].isupper()
        if bool_wrong_case_for_language:
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
            if random.random() < self.proba_wrong_question_mark_fr:
                if instruction.endswith("?"):
                    instruction = instruction[:-1].rstrip() + "?"
            # Contractions needed (au -> à l', du -> de l')
            if language_output_str[0] in "aeiouyAEIOUY":  # only when the word starts with a vowel
                instruction = re.sub(r"\bau <language_to>", r"à l'" + language_output_str, instruction)
                instruction = re.sub(r"\bdu <language_to>", r"de l'" + language_output_str, instruction)
            # Feminine words ("version française / anglaise")
            instruction = re.sub(r"\b(version) <language_to>", r"\1 " + language_output_str + "e", instruction)
            instruction = re.sub(r"\b(version) <language_from>", r"\1 " + language_input_str + "e", instruction)

        # - Then the general substitutions
        instruction = re.sub("<language_from>", language_input_str, instruction)
        instruction = re.sub("<language_to>", language_output_str, instruction)

        # Instruction ending
        if instruction.endswith("?"):
            instruction = instruction[:-1] + ending_question_mark
        elif instruction.endswith("."):
            instruction = instruction[:-1] + ending_point
        else:
            instruction += ending_point

        # Sometimes in lower / upper case...
        if case_instruct == "lower":
            instruction = instruction.lower()
        elif case_instruct == "upper":
            instruction = instruction.upper()
        if case_text == "lower":
            text_input = text_input.lower()
            text_translated = text_translated.lower()
        elif case_text == "upper":
            text_input = text_input.upper()
            text_translated = text_translated.upper()

        # Keep same quotes for the response (if any)
        text_input = opening_quote + text_input + ending_quote
        text_translated = opening_quote + text_translated + ending_quote

        # Make the final instruction and response
        if bool_put_instruction_after_text:
            # First text to translate, then instruction

            if bool_translate_from_chat_history:
                # Sometimes, the text to translate is in the instruction
                last_chatbot_turn = text_input
                return self.release(
                    [
                        {"role": "assistant", "content": last_chatbot_turn},
                        {"role": "user", "content": instruction},
                        {"role": "assistant", "content": text_translated},
                    ],
                    text_input,
                    text_translated,
                    language_input,
                    language_output,
                )

            separator = separator.replace(":", "\n--\n")
            instruction = text_input + separator + instruction
        else:
            instruction = instruction + separator + text_input

        return self.release(
            [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": text_translated},
            ],
            text_input,
            text_translated,
            language_input,
            language_output,
        )

    def release(self, turns, text_input, text_translated, language_input, language_output):
        if self.verbose:
            short_text_input = f"{text_input[:20]:20}[[... ({language_input}) ...]]{text_input[-5:]:5}"
            short_text_translated = f"{text_translated[:20]:20}[[... ({language_output}) ...]]{text_translated[-5:]:5}"
            for turn in turns:
                role = turn["role"]
                content = (
                    turn["content"]
                    .replace(text_input, short_text_input)
                    .replace(text_translated, short_text_translated)
                    .replace("\n", "\\n")
                    .replace("\t", "\\t")
                )
                print(f"– {role}: {content}")
            print("---")
        return turns


_instruct_configs = {
    "croissant_aligned": InstructDataIteratorTranslation,
    "europarl_aligned": InstructDataIteratorTranslation,
}


def to_instruct_data(ds, *kargs, **kwargs):
    name = ds
    if not isinstance(ds, str):
        name = ds.name
    for config, conversion_class in _instruct_configs.items():
        config = norm_config_name(config)
        if config in name:
            return conversion_class(ds, *kargs, **kwargs)

    raise NotImplementedError(f"Dataset {name} not implemented")


def normalize_text(text, lan):
    if lan == "en":
        text = re.sub(r" ?'s ?", "'s", text)
    else:
        raise NotImplementedError(f"Normalization for {lan} not implemented")
    return text


def main_dump_parquet():
    import argparse

    import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", type=str, default="all", nargs="+", help="Datasets to dump")
    parser.add_argument("--output", type=str, default=".", help="Output directory")
    parser.add_argument(
        "--low-quality", default=False, action="store_true", help="Use historical data instead of lastly curated data"
    )
    parser.add_argument("--max_per_parquet", default=None, type=int, help="Maximum number of examples per parquet file")
    args = parser.parse_args()

    if args.datasets == ["all"]:
        args.datasets = list(_instruct_configs.keys())

    datasets = list(decompose_config(args.datasets, high_quality=not args.low_quality))
    datasets = [to_instruct_data(dataset) for dataset in datasets]
    all_datas = dict(dataset_to_key_value(dataset) for dataset in datasets)

    random.seed(1234)
    for k, data_iterator in all_datas.items():
        output_filename = os.path.join(args.output, k + ".parquet")
        if os.path.exists(output_filename):
            print(f"File {output_filename} already exists, skipping")
            continue
        print(f"Generating {output_filename}")
        datas = []
        iterator = data_iterator if not args.max_per_parquet else tqdm.tqdm(data_iterator, total=args.max_per_parquet)
        for i, messages in enumerate(iterator):
            if args.max_per_parquet and i >= args.max_per_parquet:
                break
            datas.append(messages)
        os.makedirs(args.output, exist_ok=True)
        pd.DataFrame({"messages": datas}).to_parquet(output_filename)


def dataset_to_key_value(dataset):
    if isinstance(dataset, tuple) and len(dataset) == 2:
        return dataset
    else:
        return (dataset.name.replace(":", "--"), dataset)


if __name__ == "__main__":
    main_dump_parquet()
