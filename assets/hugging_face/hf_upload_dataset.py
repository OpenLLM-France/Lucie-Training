import json
import os
import random
import shutil
import signal
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
    # Trick to dump OrderedDict
    def represent_ordereddict(dumper, data):
        return dumper.represent_dict(data.items())

    yaml.add_representer(OrderedDict, represent_ordereddict, Dumper=yaml.SafeDumper)

    # TODO: reload the original file and merge it (in case it was modified in the meantime) ?

    with open(_readme_header_file, "w") as f:
        yaml.dump(OrderedDict(_dataset_header), f, Dumper=yaml.SafeDumper, default_flow_style=False)


def sort_config_key(name):
    """
    return a tuple where the name is prefixed by an index, indicating the order (among several configs).

    example use:
        sorted(..., key=lambda config: sort_config_key(config["config_name"]))

    # 1. (0)       default
    # 2. (1 ... N) en, fr, de, ... # natural languages
    # 3. (N + 1)  multi-lingual
    # 4. (N + 2)  code
    # 5. (N + 3)  python, c++, ... # programming languages
    # 6. (N + 4)  individual subsets
    """
    _languages = ["en", "fr", "de", "es", "it"]
    N = len(_languages)
    if any(name.startswith(f"{language},") for language in ["en", "fr", "de", "es", "it"]):
        # Bilingual settings
        order_idx = N + 1
    elif name.startswith("code-"):
        # Programming languages
        order_idx = N + 3
        name = name.split("-")[1]
    else:
        order_idx = (
            {lang: i + 1 for i, lang in enumerate(_languages)}
            | {
                "default": 0,
                "code": N + 2,
            }
        ).get(name, N + 4)  # individual subsets will come at the end

    return (order_idx, name)


def to_language(name, **kwargs):
    lan, _, sub = to_language_name_subset(name, **kwargs)
    lan_type = "natural"
    if lan == "code":
        lan_type = "code"  # "programming"
        lan = sub
    return lan_type, lan


def to_source_and_id_func(name, **kwargs):
    lan, main, subset = to_language_name_subset(name, **kwargs)
    main = main.split(".")[0].split("-")[0]  # Wikipedia.en -> Wikipedia # RedPajama--fr--debug -> RedPajama

    source = None
    update_dict_func = None

    # Add default id-ing to some datasets that miss "id" field
    _other_fr = ["OtherFr", "LEGI", "AmendementsParlement", "InterventionsParlement", "QuestionsEcritesParlement"]
    if main in _other_fr + [
        "Claire",
        "ValidatedYouTube",
        "YouTube",
        "Europarl",
        "EuroparlAligned",
        "Stac",
    ]:

        def update_dict_func(x, idx, _):
            out = {}
            if subset and main not in _other_fr:
                out = {
                    "subset": subset,
                }
            if subset not in ["LEGI", "QuestionsEcritesParlement"] and main not in [
                "LEGI",
                "QuestionsEcritesParlement",
            ]:
                out["idx_row"] = idx
            return out

    if source is None:
        source = {
            ("Claire", "fr"): "Claire",  # "OpenLLM-France/Claire-Dialogue-French-0.1",
            ("Claire", "en"): "Claire",  # "OpenLLM-France/Claire-Dialogue-English-0.1",
            ("ValidatedYouTube", "fr"): "YouTube",  # "LeVoiceLab/YouTube.fr",
        }.get((main, lan))
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


class TimeoutException(Exception):
    pass


class TimeOut:
    def __init__(self, seconds, final_action=None):
        self.seconds = seconds
        self.final_action = final_action
        if self.seconds:
            assert self.seconds >= 0, f"Invalid timeout: {self.seconds}"

    def handle_timeout(self, signum, frame):
        msg = f"Timeout after {self.seconds} seconds"
        print("WARNING:" + msg)
        raise TimeoutException(msg)

    def __enter__(self):
        if self.seconds:
            signal.signal(signal.SIGALRM, self.handle_timeout)
            signal.alarm(self.seconds)

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.alarm(0)  # Disable the alarm
        if self.final_action:
            self.final_action()
        return exc_type is TimeoutException  # Suppress TimeoutException if handled


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
        "--max_time", type=float, default=(20 * 60 - 5), help="Maximum time to run, in minutes (default 19H55)"
    )
    parser.add_argument(
        "--repository", type=str, default="OpenLLM-France/Lucie-Training-Dataset", help="Hugging Face repository"
    )
    parser.add_argument(
        "--clean", default=False, action="store_true", help="Clean the parquet after they have been uploaded"
    )
    parser.add_argument(
        "--update_each", type=int, default=10, help="Update each N parquet files (to avoid too frequent uploads)"
    )
    parser.add_argument("--high-quality", default=False, action="store_true", help="Use high quality data only")
    parser.add_argument("--version", default="v1.1", help="Version of the dataset")
    parser.add_argument("--revision", default=None, help="Branch name")
    parser.add_argument("--message", type=str, default="Upload data", help="Commit message for the upload")
    args = parser.parse_args()

    revision = args.revision
    if not revision and args.version != "v1.1":
        revision = args.version

    repo_id = args.repository
    repo_url = f"https://huggingface.co/{repo_id}"

    hf_api = None

    is_branch_new = False
    revision_info = ""
    if revision:
        hf_api, _ = connect_to_huggingface(repo_id, repo_type="dataset")
        revision_info = f" (branch {revision})"
        try:
            hf_api.create_branch(repo_id, repo_type="dataset", branch=revision)
            is_branch_new = True
        except Exception as err:  # huggingface_hub.utils._errors.HfHubHTTPError ?
            print(str(err).split("\n")[-1])
        if is_branch_new:
            print(f"Create branch {revision} in {repo_url}")
        # hf_api.create_tag(repo_id, repo_type="model", revision=revision, tag=revision, tag_message=message)

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
        high_quality=args.high_quality,
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
    do_upload = repo_id and not args.collect_metadata
    must_update_readme = False

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
    parquet_finished, parquet_filename = True, None
    parquet_files_created = []
    lock_files = []
    with TimeOut((args.max_time * 60) if args.max_time else None):
        try:
            for dataset_name, dataset in progress_bar:
                dataset_name = dataset_name.strip("-")
                progress_bar.set_description(f"Processing {dataset_name}")
                dataset_pseudo = dataset_name.split("-")[0]
                if previous_pseudo != dataset_pseudo:
                    previous_pseudo = dataset_pseudo
                    random.seed(1234)  # Hack. Try to reproduce same randomness as when tokenizing

                # To yield dictionaries with metadata instead of just the text
                language_category, language = to_language(dataset_name)
                source, source_pseudo, update_dict_func = to_source_and_id_func(dataset_name)
                assert (
                    source_pseudo and language_category and language and dataset_name
                ), f"{source=} -- {source_pseudo=} -- {language=} -- {language_category=} -- {dataset_name=}"
                path_in_repo = (
                    f"data/{args.version}/{language_category}/{language}/{source_pseudo}/{dataset_name}.parquet"
                )

                progress_bar.set_description(f"Generating {path_in_repo}")

                language_configname = language.replace("-", ",")  # fr-en -> fr,en

                if source != "TheStack":  # This one is already under code-*
                    multi_lingual_configs[source] = multi_lingual_configs.get(source, {}) | {
                        language: {
                            "config_name": f"{source}-{language_configname}",
                            "data_files": [
                                {
                                    "split": "train",
                                    "path": f"data/{language_category}/{language}/{source_pseudo}/*.parquet",
                                }
                            ],
                        }
                    }

                config_names = [c["config_name"] for c in _dataset_header["configs"]]
                for config in [
                    {
                        "config_name": source,
                        "data_files": [
                            {"split": "train", "path": f"data/{language_category}/*/{source_pseudo}/*.parquet"}
                        ],
                    },
                    {
                        "config_name": language_configname
                        if (language_category == "natural")
                        else f"{language_category}-{language_configname}",
                        "data_files": [{"split": "train", "path": f"data/{language_category}/{language}/*/*.parquet"}],
                    },
                ] + (
                    list(multi_lingual_configs.get(source).values())
                    if len(multi_lingual_configs.get(source, [])) > 1
                    else []
                ):
                    if config["config_name"] not in config_names:
                        _dataset_header["configs"].append(config)
                        _dataset_header["configs"] = sorted(
                            _dataset_header["configs"], key=lambda x: sort_config_key(x["config_name"])
                        )
                        must_update_readme = True

                if must_update_readme:
                    dump_dataset_config()

                parquet_filename = os.path.join(args.folder, path_in_repo)
                os.makedirs(os.path.dirname(parquet_filename), exist_ok=True)

                lock_file = parquet_filename + ".lock"
                if do_upload:
                    if os.path.exists(lock_file):
                        continue
                    with open(lock_file, "w") as f:
                        f.write("lock")
                    lock_files.append(lock_file)

                dataset.SetYieldMetadata(
                    uniformize_metadata=args.uniformize_metadata,
                    extra_metadata=dict(
                        source=source,
                        language=language,
                    ),
                    update_dict_func=update_dict_func,
                )

                metadatas[source] = metadatas.get(source, {})
                examples[source] = examples.get(source, {})

                all_data = {}
                if metadata_fields:
                    for k in metadata_fields:
                        all_data[k] = []

                has_data = False
                parquet_finished = do_upload and os.path.isfile(parquet_filename)
                if parquet_finished:
                    print(f"Warning: Using existing parquet file {parquet_filename}")
                else:
                    for i, sample in enumerate(tqdm.tqdm(dataset, desc=f"Generating {parquet_filename}")):
                        has_data = True
                        assert isinstance(sample, dict), f"Sample is not a dictionary: {type(sample)}"
                        assert "text" in sample and isinstance(sample["text"], str)

                        # Update metadata
                        if args.collect_metadata:
                            # Update types
                            metadatas[source] = get_union(
                                [
                                    {k: get_type(sample[k]) for k in sorted(sample.keys()) if k != "text"},
                                    metadatas[source],
                                ],
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
                                if v is None:
                                    v = _DEFAULT_VALUE
                                all_data[k].append(v)
                            for k in all_data:
                                if k not in sample:
                                    all_data[k].append(_DEFAULT_VALUE)

                    if not has_data:
                        print(f"Dataset {dataset_name} has no data")
                        continue
                        # raise RuntimeError(f"Dataset {dataset_name} has no data")

                    if do_upload:
                        # Dump parquet file
                        all_data = pd.DataFrame(all_data)
                        all_data.to_parquet(parquet_filename)

                    parquet_finished = True

                del all_data

                if do_upload:
                    parquet_files_created.append(parquet_filename)
                    if len(parquet_files_created) >= args.update_each:
                        parquet_files_created = list(set(parquet_files_created))

                        # Dump to Hugging Face
                        if hf_api is None:
                            hf_api, _ = connect_to_huggingface(repo_id, repo_type="dataset")

                        common_path = (
                            os.path.commonpath(parquet_files_created)
                            if len(parquet_files_created) > 1
                            else os.path.dirname(parquet_files_created[0])
                        )

                        if len(parquet_files_created) == 1:
                            hf_api.upload_file(
                                path_or_fileobj=parquet_filename,
                                path_in_repo=path_in_repo,
                                commit_message=args.message if args.message else f"Upload {source}",
                                repo_id=repo_id,
                                repo_type="dataset",
                                revision=revision,
                            )
                        else:
                            hf_api.upload_folder(
                                folder_path=common_path,
                                path_in_repo=os.path.relpath(common_path, args.folder),
                                commit_message=args.message if args.message else "Upload data",
                                ignore_patterns=["*.lock"],
                                repo_id=repo_id,
                                repo_type="dataset",
                                revision=revision,
                            )

                        if args.clean:
                            for f in parquet_files_created:
                                if os.path.exists(f):
                                    os.remove(f)

                        parquet_files_created = []
                        lock_files = []

        except (Exception, KeyboardInterrupt) as err:
            if not parquet_finished and os.path.exists(parquet_filename):
                os.remove(parquet_filename)
            for f in lock_files:
                if os.path.exists(f):
                    os.remove(f)
            raise err

    if len(parquet_files_created):
        # Dump the last ones

        if hf_api is None:
            hf_api, _ = connect_to_huggingface(repo_id, repo_type="dataset")

        common_path = (
            os.path.commonpath(parquet_files_created)
            if len(parquet_files_created) > 1
            else os.path.dirname(parquet_files_created[0])
        )

        hf_api.upload_folder(
            folder_path=common_path,
            path_in_repo=os.path.relpath(common_path, args.folder),
            commit_message=args.message if args.message else "Upload data",
            ignore_patterns=["*.lock"],
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
        )

        if args.clean:
            for f in parquet_files_created:
                if os.path.exists(f):
                    os.remove(f)

    # # TODO ??? for now we don't automatically update the README.md (too dangerous)
    # if must_update_readme:
    if False:
        if hf_api is None:
            hf_api, _ = connect_to_huggingface(repo_id, repo_type="dataset")

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
            commit_message="Update README.md",
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
        )

        os.remove(tmp_file)
