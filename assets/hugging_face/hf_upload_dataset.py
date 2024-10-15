import json
import os
import random
import shutil
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path

import pandas as pd
import tqdm
import yaml

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, "tokenization"))

from tokenizer_apply import dataset_to_key_value  # noqa: E402

from assets.compile_stats import to_language_name_subset  # noqa: E402 Module level import not at top of file
from assets.hugging_face.hf_upload_model import connect_to_huggingface  # noqa: E402
from data import decompose_datasets, get_datasets  # noqa: E402

_UNION_KEY = "__UNION__"
_DEFAULT_VALUE = ""  # Putting None cause some cast issues for datasets without the field

wd = Path(__file__).parent.resolve()

_readme_file_main = wd / "README_dataset.md"
_readme_header_file = wd / "README_dataset_header.yaml"
for fn in [_readme_file_main, _readme_header_file]:
    assert fn.exists(), f"File not found at {fn}"

with open(_readme_header_file) as f:
    _dataset_header = OrderedDict(yaml.safe_load(f))


def dump_dataset_config():
    def represent_ordereddict(dumper, data):
        return dumper.represent_dict(data.items())

    yaml.add_representer(OrderedDict, represent_ordereddict, Dumper=yaml.SafeDumper)
    with open(_readme_header_file, "w") as f:
        yaml.dump(OrderedDict(_dataset_header), f, Dumper=yaml.SafeDumper, default_flow_style=False)


def to_language(name, **kwargs):
    lan, _, __ = to_language_name_subset(name, **kwargs)
    return lan


def to_source_and_id_func(name, **kwargs):
    lan, main, subset = to_language_name_subset(name, **kwargs)
    main = main.split(".")[0].split("-")[0]  # Wikipedia.en -> Wikipedia # RedPajama--fr--debug -> RedPajama

    source = None
    update_dict_func = None

    # Add default id-ing to some datasets that miss "id" field
    if main in ["Claire", "ValidatedYoutube", "OtherFr", "Europarl", "EuroparlAligned", "Stac"]:

        def update_dict_func(x, idx, _):
            out = {}
            if subset:
                out = {
                    "subset": subset,
                }
            out["idx_row"] = idx
            return out

    if source is None:
        source = {
            ("Claire", "fr"): "Claire",  # "OpenLLM-France/Claire-Dialogue-French-0.1",
            ("Claire", "en"): "Claire",  # "OpenLLM-France/Claire-Dialogue-English-0.1",
            ("ValidatedYouTube", "fr"): "YouTube",  # "LeVoiceLab/YouTube.fr",
        }.get((main, lan))
    if source is None:
        source = {
            ("OtherFr", "questions_ecrites_parlement"): "QuestionsEcritesParlement",
            ("OtherFr", "interventions_parlement"): "InterventionsParlement",
            ("OtherFr", "LEGI"): "LEGI",
            ("OtherFr", "amendements_parlement"): "AmendementsParlement",
        }.get((main, subset))
    if main == "Wikiother":
        source = subset.split(":")[0]
        source = source[0].upper() + source[1:]
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
            "Stac": "XXX",
            "TheStack": "XXX",
            "Theses": "XXX",
            "Wikiother": "XXX",
            "Wikipedia": "XXX",
        }.get(main)
    if source is None:
        source = main
        # raise NotImplementedError(f"Missing source for {name=} ({lan=}, {main=}, {subset=})")
    if source == "XXX":
        source = main
    return source, source, update_dict_func


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
    parser.add_argument("--folder", type=str, default="tmp_upload", help="Output folder")
    parser.add_argument(
        "--repository", type=str, default="OpenLLM-France/Lucie-Training-Dataset", help="Hugging Face repository"
    )
    parser.add_argument(
        "--clean", default=False, action="store_true", help="Clean the parquet after they have been uploaded"
    )
    args = parser.parse_args()

    metadata_fields = {
        "source": str,
        "id": str,
        "language": str,
        "date": str,
        "author": str,
        "url": str,
        "title": str,
        "extra": str,
        "quality_signals": str,
    }

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
    multi_lingual_configs = {}
    examples = {}
    metadatas[_UNION_KEY] = {}
    examples[_UNION_KEY] = {}
    args.collect_metadata_examples = args.collect_metadata
    args.collect_metadata = (
        (os.path.splitext(args.collect_metadata)[0] + "_types.json") if args.collect_metadata else None
    )
    do_upload = args.repository and not args.collect_metadata

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
    hf_api = None
    for dataset_name, dataset in progress_bar:
        dataset_pseudo = dataset_name.split("-")[0]
        if previous_pseudo != dataset_pseudo:
            previous_pseudo = dataset_pseudo
            random.seed(1234)  # Hack. Try to reproduce same randomness as when tokenizing

        # To yield dictionaries with metadata instead of just the text
        language = to_language(dataset_name)
        source, source_pseudo, update_dict_func = to_source_and_id_func(dataset_name)
        assert source_pseudo and language and dataset_name
        path_in_repo = f"{source_pseudo}/{language}/{dataset_name}.parquet"

        multi_lingual_configs[source_pseudo] = multi_lingual_configs.get(source_pseudo, {}) | {
            language: {
                "config_name": f"{source_pseudo}/{language}",
                "data_files": [{"split": "train", "path": f"data/{source_pseudo}/{language}/*.parquet"}],
            }
        }

        config_names = [c["config_name"] for c in _dataset_header["configs"]]
        for config in [
            {
                "config_name": source_pseudo,
                "data_files": [{"split": "train", "path": f"data/{source_pseudo}/*/*.parquet"}],
            },
            {"config_name": language, "data_files": [{"split": "train", "path": f"data/*/{language}/*.parquet"}]},
        ] + (
            multi_lingual_configs.get(source_pseudo).values() if len(multi_lingual_configs[source_pseudo]) > 1 else []
        ):
            if config["config_name"] not in config_names:
                _dataset_header["configs"].append(config)
                dump_dataset_config()

        parquet_filename = os.path.join(args.folder, path_in_repo)
        os.makedirs(os.path.dirname(parquet_filename), exist_ok=True)

        lock_file = parquet_filename + ".lock"
        if os.path.exists(lock_file):
            continue
        with open(lock_file, "w") as f:
            f.write("lock")

        try:
            dataset.SetYieldMetadata(
                uniformize_metadata=args.uniformize_metadata,
                extra_metadata=dict(
                    source=source,
                    language=language,
                ),
                update_dict_func=update_dict_func,
            )

            # if args.collect_metadata and source in metadatas:
            #     continue
            progress_bar.set_description(f"Processing {dataset_name}...")

            metadatas[source] = metadatas.get(source, {})
            examples[source] = examples.get(source, {})

            all_data = {}
            if metadata_fields:
                for k in metadata_fields:
                    all_data[k] = []

            has_data = False
            for i, sample in enumerate(dataset):
                has_data = True
                assert isinstance(sample, dict), f"Sample is not a dictionary: {type(sample)}"
                assert "text" in sample and isinstance(sample["text"], str)

                # Update metadata
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

                # Add sample data to (parquet) dataset
                if do_upload:
                    if not all_data:
                        all_data = {k: [] for k in sample.keys()}
                    for k, v in sample.items():
                        if k not in all_data:
                            assert all_data
                            num_samples = len(all_data[list(all_data.keys())[0]])
                            all_data[k] = [_DEFAULT_VALUE] * num_samples
                        all_data[k].append(v)
                    for k in all_data:
                        if k not in sample:
                            all_data[k].append(_DEFAULT_VALUE)

            if not has_data:
                raise RuntimeError(f"Dataset {dataset_name} has no data")

            if do_upload:
                # Dump parquet file
                pd.DataFrame(all_data).to_parquet(parquet_filename)

                # Dump to Hugging Face
                if hf_api is None:
                    hf_api, _ = connect_to_huggingface(args.repository, repo_type="dataset")

                hf_api.upload_file(
                    path_or_fileobj=parquet_filename,
                    path_in_repo=f"data/{path_in_repo}",
                    commit_message=f"Upload {os.path.splitext(path_in_repo)[0]}",
                    repo_id=args.repository,
                    repo_type="dataset",
                    revision=None,
                )

                if args.clean:
                    os.remove(parquet_filename)

        except Exception as err:
            os.remove(lock_file)
            raise err

        if do_upload:
            # Create the README.md file
            readme_content = "---\n"
            with open(_readme_header_file) as f:
                readme_content += f.read().strip() + "\n"
            readme_content += "---\n"
            with open(_readme_file_main) as f:
                readme_content += "\n" + f.read().strip() + "\n"
            tmp_file = os.path.join(tempfile.gettempdir(), "README.md")
            with open(tmp_file, "w") as f:
                f.write(readme_content)

            hf_api.upload_file(
                path_or_fileobj=tmp_file,
                path_in_repo="README.md",
                commit_message="Upload README.md",
                repo_id=args.repository,
                repo_type="dataset",
                revision=None,
            )

            os.remove(tmp_file)
