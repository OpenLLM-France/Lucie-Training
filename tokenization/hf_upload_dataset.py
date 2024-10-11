import json
import os
import random
import shutil
import sys
import tempfile

import tqdm
from tokenizer_apply import dataset_to_key_value

from data import decompose_datasets, get_datasets

_UNION_KEY = "__UNION__"

# TODO put this read_stats_datasets code somewhere else
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
asset_folder = os.path.join(parent_dir, "assets")
sys.path.append(asset_folder)
from compile_stats import to_language_name_subset  # noqa: E402 Module level import not at top of file


def to_language(name, **kwargs):
    lan, _, __ = to_language_name_subset(name, **kwargs)
    return lan


def to_source_and_id_func(name, **kwargs):
    lan, main, subset = to_language_name_subset(name, **kwargs)
    main = main.split(".")[0].split("-")[0]  # Wikipedia.en -> Wikipedia # RedPajama--fr--debug -> RedPajama

    source = None
    id_func = None

    # Add default id-ing to some datasets that miss "id" field
    if main in ["Claire", "ValidatedYoutube", "OtherFr", "Europarl", "EuroparlAligned"]:

        def id_func(x, idx, _):
            return f"{subset or main}/{idx}"

    if source is None:
        source = {
            ("Claire", "fr"): "OpenLLM-France/Claire-Dialogue-French-0.1",
            ("Claire", "en"): "OpenLLM-France/Claire-Dialogue-English-0.1",
            ("ValidatedYoutube", "fr"): "LeVoiceLab/Youtube.fr",
        }.get((main, lan))
    if source is None:
        source = {
            ("OtherFr", "questions_ecrites_parlement"): "questions_ecrites_parlement",
            ("OtherFr", "interventions_parlement"): "interventions_parlement",
            ("OtherFr", "LEGI"): "LEGI",
            ("OtherFr", "amendements_parlement"): "amendements_parlement",
            ("Wikiother", "wikisource"): "Wikisource",
            ("Wikiother", "wiktionary"): "Wiktionary",
        }.get((main, subset))
    if source is None:
        source = {
            "AmericanStories": "XXX",
            "CroissantAligned": "XXX",
            "DiscoursPublics": "XXX",
            "Europarl": "XXX",
            "EuroparlAligned": "XXX",
            "Eurovoc": "XXX",
            "FineWebEdu": "XXX",
            "GallicaMonographies": "XXX",
            "GallicaPress": "XXX",
            "Gutenberg": "XXX",
            "HAL": "XXX",
            "MathPile": "XXX",
            "OpenData": "XXX",
            "OpenEdition": "XXX",
            "PeS2o": "XXX",
            "Persee": "XXX",
            "RedPajama": "XXX",
            "TheStack": "XXX",
            "Theses": "XXX",
            "ValidatedYouTube": "XXX",
            "Wikiother": "XXX",
            "Wikipedia": "XXX",
        }.get(main)
    if source is None:
        raise NotImplementedError(f"Missing source for {name=} ({lan=}, {main=}, {subset=})")
    if source == "XXX":
        source = main
    return source, id_func


def get_type(v):
    if v is None:
        return None
    t = type(v)
    if t == dict:
        return {k: get_type(v) for k, v in sorted(v.items())}
    if t in (list, tuple):
        if len(v):
            return f"{str(t)}[{get_type(v[0])}]"
        else:
            return f"{str(t)}[]"
    return str(t)


def get_example_preview(v, enforce_dict=False, max_string_length=None):
    if enforce_dict and isinstance(v, str) and v.startswith("{"):
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            print(f"WARNING: Failed to parse {v}")
            pass
    if isinstance(v, list):
        if len(v) > 3:
            v = v[:1] + ["…"] + v[-1:]
        v = [get_example_preview(x, enforce_dict=enforce_dict) for x in v]
        return v
    if isinstance(v, dict):
        v = {k: get_example_preview(v, enforce_dict=enforce_dict) for k, v in sorted(v.items())}
        return v
    if isinstance(v, str):
        if max_string_length and len(v) > max_string_length:
            v = v[: max_string_length // 2] + "…" + v[-max_string_length // 2 :]
        return v
    if isinstance(v, (int, float)):
        return v
    if v is None:
        return None
    return get_example_preview(str(v))


def get_union(metadatas, desc=None):
    union = {}
    for m in metadatas:
        for k, v in m.items():
            if k not in union:
                union[k] = v
            elif union[k] != v:
                # First value
                if v is None:
                    continue
                if union[k] is None:
                    union[k] = v
                    continue

                if isinstance(union[k], list) and not isinstance(v, list):
                    if v in union[k]:
                        continue
                    new_union_k = union[k] + [v]
                elif not isinstance(union[k], list) and isinstance(v, list):
                    if union[k] in v:
                        union[k] = v
                        continue
                    else:
                        new_union_k = [union[k]] + v
                elif isinstance(union[k], list) and isinstance(v, list):
                    new_union_k = union[k] + [x for x in v if x not in union[k]]
                else:
                    new_union_k = [union[k], v]
                if isinstance(new_union_k, list):
                    # Make unique and sort
                    #  not using the following because both lines fail on dictionaries
                    #  >   new_union_k = list(set(new_union_k))
                    #  >   new_union_k = sorted(new_union_k)
                    new_union = []
                    for x in new_union_k:
                        if x not in new_union:
                            new_union.append(x)
                    new_union_k = sorted(new_union, key=lambda x: str(x))

                    if new_union_k != union[k]:
                        print(f"Warning ({desc}): {new_union_k} for {k}")
                union[k] = new_union_k
    # Sort keys
    union = {k: union[k] for k in sorted(union.keys())}
    return union


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect all text with metadata.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--datasets", type=str, default="all", help="Datasets", nargs="+")
    parser.add_argument("--collect_metadata", type=str, help="json file to store all metadata")
    parser.add_argument("--uniformize_metadata", action="store_true", default=False, help="Uniformize metadata")
    args = parser.parse_args()

    all_datas = get_datasets(
        args.datasets,
        max_parquet_files=1 if args.collect_metadata else None,
        force_include_all_metadata=True,
    )

    all_datas = dict(
        dataset_to_key_value(dataset)
        for dataset in decompose_datasets(
            all_datas, parquet_level=not args.collect_metadata, return_json_file_if_possible=False
        )
    )

    metadatas = {}
    examples = {}
    metadatas[_UNION_KEY] = {}
    examples[_UNION_KEY] = {}
    args.collect_metadata_examples = (
        (os.path.splitext(args.collect_metadata)[0] + "_examples.json") if args.collect_metadata else None
    )
    if args.collect_metadata:
        if os.path.exists(args.collect_metadata):
            with open(args.collect_metadata) as f:
                metadatas.update(json.load(f))
    if args.collect_metadata_examples:
        if os.path.exists(args.collect_metadata_examples):
            with open(args.collect_metadata_examples) as f:
                examples.update(json.load(f))

    tmpfile = tempfile.mktemp(suffix=".json")

    def dump_metadata(metadatas, examples):
        metadatas[_UNION_KEY] = get_union(list(metadatas.values()) + [metadatas.get(_UNION_KEY, {})], desc=_UNION_KEY)
        # Sort keys (UNION at first)
        examples = {k: examples[k] for k in sorted(examples.keys(), key=lambda x: "" if x == _UNION_KEY else x)}
        metadatas = {k: metadatas[k] for k in sorted(metadatas.keys(), key=lambda x: "" if x == _UNION_KEY else x)}
        with open(tmpfile, "w") as f:
            json.dump(metadatas, f, indent=2, ensure_ascii=False)
        shutil.move(tmpfile, args.collect_metadata)
        with open(tmpfile, "w") as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)
            # import pprint
            # print(pprint.pformat(examples), file=f)
        shutil.move(tmpfile, args.collect_metadata_examples)

    progress_bar = tqdm.tqdm(all_datas.items())
    previous_pseudo = None
    for dataset_name, dataset in progress_bar:
        dataset_pseudo = dataset_name.split("-")[0]
        if previous_pseudo != dataset_pseudo:
            previous_pseudo = dataset_pseudo
            random.seed(1234)  # Hack. Try to reproduce same randomness as when tokenizing

        # To yield dictionaries with metadata instead of just the text
        language = to_language(dataset_name)
        source, id_func = to_source_and_id_func(dataset_name)
        dataset.SetYieldMetadata(
            uniformize_metadata=args.uniformize_metadata,
            extra_metadata=dict(
                source=source,
                language=language,
            ),
            id_func=id_func,
        )

        # if args.collect_metadata and source in metadatas:
        #     continue
        progress_bar.set_description(f"Processing {dataset_name}...")

        metadatas[source] = metadatas.get(source, {})
        examples[source] = examples.get(source, {})

        has_data = False
        for i, sample in enumerate(dataset):
            has_data = True
            assert isinstance(sample, dict), f"Sample is not a dictionary: {type(sample)}"
            assert "text" in sample and isinstance(sample["text"], str)
            if args.collect_metadata:
                # Update types
                metadatas[source] = get_union(
                    [{k: get_type(sample[k]) for k in sorted(sample.keys()) if k != "text"}, metadatas[source]],
                    desc=dataset_name,
                )
                # Update examples
                for k, v in sample.items():
                    if v is None or k in ["text"]:
                        continue
                    if k not in examples[source]:
                        examples[source][k] = get_example_preview(
                            v, enforce_dict=False
                        )  # True # args.uniformize_metadata)
                    if k not in examples[_UNION_KEY]:
                        examples[_UNION_KEY][k] = get_example_preview(v)
                dump_metadata(metadatas, examples)
                if None not in metadatas[source].values() or i > 100:
                    break

        if not has_data:
            raise RuntimeError(f"Dataset {dataset_name} has no data")
