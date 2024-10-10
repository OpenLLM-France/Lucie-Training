import json
import os
import random
import shutil
import tempfile

import tqdm
from tokenizer_apply import dataset_to_key_value

from data import decompose_datasets, get_datasets


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
    if args.collect_metadata and os.path.exists(args.collect_metadata):
        with open(args.collect_metadata) as f:
            metadatas = json.load(f)

    tmpfile = tempfile.mktemp(suffix=".json")

    def dump_metadata(metadatas):
        # Sort keys
        metadatas = {k: metadatas[k] for k in sorted(metadatas.keys()) if k != "UNION"}
        # Add union at last
        metadatas["UNION"] = get_union(list(metadatas.values()) + [metadatas.get("UNION", {})], desc="UNION")
        with open(tmpfile, "w") as f:
            json.dump(metadatas, f, indent=2)
        shutil.move(tmpfile, args.collect_metadata)

    progress_bar = tqdm.tqdm(all_datas.items())
    previous_pseudo = None
    for dataset_name, dataset in progress_bar:
        dataset_pseudo = dataset_name.split("-")[0]
        if previous_pseudo != dataset_pseudo:
            previous_pseudo = dataset_pseudo
            random.seed(1234)  # Try to reproduce same randomness as when tokenizing
        # if args.collect_metadata and dataset_pseudo in metadatas:
        #     continue
        progress_bar.set_description(f"Processing {dataset_name}...")

        metadatas[dataset_pseudo] = metadatas.get(dataset_pseudo, {})

        # Hack to return dictionary, not just the text
        dataset.key = None

        for i, sample in enumerate(dataset):
            assert isinstance(sample, dict)
            assert "text" in sample and isinstance(sample["text"], str)
            if args.collect_metadata:
                metadatas[dataset_pseudo] = get_union(
                    [{k: get_type(sample[k]) for k in sorted(sample.keys()) if k != "text"}, metadatas[dataset_pseudo]],
                    desc=dataset_name,
                )
                dump_metadata(metadatas)
                if None not in metadatas[dataset_pseudo].values() or i > 100:
                    break
