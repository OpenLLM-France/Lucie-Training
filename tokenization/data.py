import itertools
import json
import os
import re
import time

import datasets
import tqdm

_folder = os.path.dirname(os.path.realpath(__file__))
_asset_folder = os.path.join(os.path.dirname(_folder), "assets")


def get_datasets(config_names=None, high_quality=False, streaming=True, **kwargs):
    if config_names in [None, "all"]:
        config_names = get_all_config_names()
    if isinstance(config_names, str):
        config_names = [config_names]

    for config_name in config_names:
        dataset = DataIterator(config_name, high_quality=high_quality, streaming=streaming, **kwargs)
        yield from decompose_datasets(dataset)


def decompose_datasets(dataset, max_samples=None):
    num_samples = 0
    num_chars = 0
    for sample in dataset:
        print("NOCOMMIT", sample)
        if num_samples >= 100:
            break
        assert isinstance(sample, str), f"Invalid type for {sample}"
        num_samples += 1
        num_chars += len(sample)
    avg_num_chars = num_chars / num_samples

    if max_samples is None:
        max_samples = 2_000_000_000 / avg_num_chars

    print(f"Decomposing dataset {dataset.name} with {max_samples} samples")

    ds = dataset.hf_dataset
    previous_sample = None
    while True:
        new_ds = ds.take(max_samples)
        ds = ds.skip(max_samples)
        print("Finished one dataset")
        yield DataIterator(new_ds, streaming=dataset.streaming)
        new_sample = None
        try:
            for x in ds:
                new_sample = x
                break
        except ValueError:
            break
        if new_sample == previous_sample and new_sample is not None:
            break
        previous_sample = new_sample
        ds = new_ds


def get_subsets(hf_dataset):
    subsets = set()
    for sample in hf_dataset:
        subset = _get_field(sample, "subset")
        if subset:
            subsets.add(subset)
    return sorted(subsets)


def _get_field(sample, field, in_="extra"):
    if in_:
        sample = sample[in_]
    if isinstance(sample, str):
        try:
            sample = json.loads(sample)
        except json.JSONDecodeError:
            return None
    assert isinstance(sample, dict), f"Invalid type for {sample} ({type(sample)})"
    return sample.get(field)


def _filter_sample_by_subset(subset, sample):
    return _get_field(sample, "subset") == subset


def _filter_sample_by_index_range(index_min, index_max, index):
    return index_min <= index <= index_max


def tokenizer_datasets(train=True, factor=1):
    raise NotImplementedError("Not re-implemented since refactoring yet")


def get_all_config_names():
    config_names = list(datasets.load_dataset_builder("OpenLLM-France/Lucie-Training-Dataset").builder_configs)

    def include_config_name(all_names, name):
        _languages = ["fr", "en", "de", "es", "it"]

        # Add language combinations
        _languages += [
            ",".join(combo) for r in range(1, len(_languages) + 1) for combo in itertools.permutations(_languages, r)
        ]

        # Try to deliver subsets if possible
        if name in [
            "default",
            "natural",
            "code",
            "PeS2o",  # PeS2o-s2ag, ...
            "Pile",  # Pile-DM_Mathematics, ...
            "TheStack",  # code-c#, ...
        ]:
            return False

        # Skip language specific configs
        if name in _languages:
            return False

        # Skip multi-lingual configs
        for lan in _languages:
            if name + "-" + lan in all_names:
                return False
        return True

    return [name for name in config_names if include_config_name(config_names, name)]


class DataIterator:
    def __init__(self, config_name="default", high_quality=False, streaming=True, **kwargs):
        revision = "v1.2" if high_quality else "v1.1"

        if isinstance(config_name, str):
            hf_dataset = datasets.load_dataset(
                "OpenLLM-France/Lucie-Training-Dataset",
                config_name,
                revision=revision,
                streaming=streaming,
                split="train",
                **kwargs,
            )
        else:
            hf_dataset = config_name

        self.hf_dataset = hf_dataset
        self.dataset_iter = hf_dataset.__iter__()
        self.streaming = streaming

    def __iter__(self):
        return self

    def __next__(self):
        sample = next(self.dataset_iter)
        return sample["text"]

    def __len__(self):
        try:
            return len(self.hf_dataset)
        except TypeError:
            return 0

    @property
    def name(self) -> dict:
        return self.hf_dataset.config_name


########################################
# Main


def main():
    import argparse
    import shutil

    parser = argparse.ArgumentParser(
        description="Test the data iterators and print statistics about datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset",
        nargs="*",
        default=["all"],
        help="Which dataset to test",
    )
    parser.add_argument("--high-quality", default=False, action="store_true", help="Use high quality data only")
    parser.add_argument(
        "--folder",
        type=str,
        default=os.path.join(_asset_folder, "stats_raw"),
        # default=None,
        help="Folder to dump some example data into",
    )
    parser.add_argument(
        "--ignore_if_exists",
        action="store_true",
        default=False,
        help="Skip if stat is already computed",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=10,
        help="Number of pages to dump as examples (when --folder is specified)",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Maximum number of samples to iterate on",
    )
    parser.add_argument(
        "--only_dump_examples",
        action="store_true",
        default=False,
        help="Only dump some examples",
    )
    parser.add_argument(
        "--long_examples",
        action="store_true",
        default=False,
        help="Only dump long examples (more than 50k words)",
    )
    args = parser.parse_args()

    if args.folder:
        os.makedirs(args.folder, exist_ok=True)
        shutil.copy2(__file__, os.path.join(args.folder, os.path.basename(__file__)))

    def remove_common_prefix(main, sub):
        common_prefix = os.path.commonprefix([main, sub])
        return sub[len(common_prefix) :]

    def update_stats(global_stats, stats):
        for k, v in stats.items():
            if k not in global_stats:
                global_stats[k] = 0
            global_stats[k] += v

    all_datasets = [get_datasets(name, high_quality=args.high_quality) for name in args.dataset]
    print(all_datasets[0])
    all_datasets = [list(decompose_datasets(it)) for sublist in all_datasets for it in sublist]
    print(all_datasets[0])
    all_datasets = [it for sublist in all_datasets for it in sublist]
    print(all_datasets[0])  # NOCOMMIT

    # Early checks to avoid failure in the middle
    for it in all_datasets:
        assert it.name, f"Missing name for {it}"

    for it in all_datasets:
        num_examples = args.num_examples
        name = it.name
        name_slug = simple_slugify(name)
        main_prefix_example_files = None
        main_stat_filename = os.path.join(args.folder, f"stats_{name_slug}.json") if args.folder else None
        if main_stat_filename and os.path.isfile(main_stat_filename) and args.only_dump_examples:
            stats = json.load(open(main_stat_filename, encoding="utf8"))
            num_billion_words = stats["num words"] / 1_000_000_000
            main_prefix_example_files = f"{num_billion_words:06.3f}B_{name_slug}"
        # elif args.only_dump_examples:
        #     raise RuntimeError(f"Missing main stat file {main_stat_filename}")

        its = [it]
        global_stats = None

        try:
            max_num_examples_per_subset = num_examples  # / len(its)
            for subset in its:
                subname = subset.name
                num_examples = int(max_num_examples_per_subset)
                if num_examples == 0 and any(s in subname for s in ("tex", "python")):
                    num_examples = 2
                if "other" in name.lower():
                    num_examples = args.num_examples
                if num_examples == 0 and args.only_dump_examples:
                    continue
                print(f"* {subname}")
                if main_prefix_example_files:
                    suffix = remove_common_prefix(name_slug, simple_slugify(subname))
                    prefix_example_files = f"{main_prefix_example_files}{suffix}"
                else:
                    prefix_example_files = None
                stats = test_iterator(
                    subset,
                    folder=args.folder,
                    name=subname,
                    ignore_if_exists=args.ignore_if_exists,
                    num_examples=num_examples,
                    only_dump_examples=args.only_dump_examples,
                    prefix_example_files=prefix_example_files,
                    max_examples=args.max_examples,
                    long_examples=args.long_examples,
                )
                if args.only_dump_examples:
                    continue
                print(json.dumps(stats, indent=4))

                if global_stats is not None:
                    update_stats(global_stats, stats)
        except Exception as err:
            raise RuntimeError(f"Error while iterating on '{subname}'") from err

        if args.only_dump_examples:
            continue

        if global_stats is not None:
            print(f"* {name}")
            print(json.dumps(global_stats, indent=4))
            if args.folder:
                json.dump(
                    global_stats,
                    open(main_stat_filename, "w", encoding="utf8"),
                    indent=2,
                    ensure_ascii=False,
                )


########################################
# Test Helpers


def test_iterator(
    it,
    folder=None,
    name="",
    ignore_if_exists=False,
    num_examples=0,
    only_dump_examples=False,
    prefix_example_files=None,
    max_examples=None,
    long_examples=False,
):
    name_slug = simple_slugify(name)
    if prefix_example_files is None:
        prefix_example_files = name_slug
    stats = None
    if folder:
        stat_filename = os.path.join(folder, f"stats_{name_slug}.json")
        if os.path.isfile(stat_filename):
            stats = json.load(open(stat_filename, encoding="utf8"))
            if len(stats):
                if ignore_if_exists and not only_dump_examples:
                    print(f"Skipping {name_slug} (already computed)")
                    return stats
                # num_billion_words = stats["num words"] / 1_000_000_000
                # to_insert = f"{num_billion_words:06.3f}B"
                # if "--" in prefix_example_files:
                #     prefix_example_files = prefix_example_files.replace("--", "--" + to_insert + "_", 1)
                # else:
                #     prefix_example_files += "--" + to_insert
        elif ignore_if_exists:
            # Create an empty file to avoid recomputing
            json.dump({}, open(stat_filename, "w", encoding="utf8"))
    print(f"Computing stats for {name_slug}...")
    tic = time.time()
    num_docs = 0
    num_words = None
    num_chars = None
    num_dumped = 0
    num_samples = len(it)
    for text in tqdm.tqdm(it, total=num_samples if num_samples else -1):
        if max_examples and num_dumped >= max_examples:
            break
        num_docs += 1

        # Accumulate number of words and characters
        if isinstance(text, str):
            if num_words is None:
                num_words = 0
                num_chars = 0
            nw = len(text.split())
            num_words += nw
            num_chars += len(text)
        else:
            assert isinstance(text, dict)
            if num_words is None:
                num_words = {}
                num_chars = {}
            nw = 0
            for k, v in text.items():
                if isinstance(v, list):
                    v = " ".join(v)
                assert isinstance(v, str), f"Invalid type for {k}: {v}"
                if k not in num_words:
                    num_words[k] = 0
                    num_chars[k] = 0
                nwi = len(v.split())
                nw += nwi
                num_words[k] += nwi
                num_chars[k] += len(v)

        if num_dumped < num_examples and folder and (not long_examples or nw > 50_000):
            example_folder = os.path.join(folder, "long_examples" if long_examples else "examples")
            os.makedirs(example_folder, exist_ok=True)
            filename = os.path.join(example_folder, f"{prefix_example_files}")
            if num_examples > 1:
                filename += f"_{num_dumped:02d}"
            filename += ".txt"
            if num_dumped == 0:
                print(f"Dumping {filename}")
            with open(filename, "w", encoding="utf8") as f:
                f.write(text + "\n")
            num_dumped += 1
        elif num_dumped >= num_examples and only_dump_examples:
            break
    if only_dump_examples:
        return {}
    if num_docs <= 0:
        raise RuntimeError("No page found, or iterations stopped before completion (stats are not full)")
    toc = time.time()
    stats = {
        "time to iterate (sec)": toc - tic,
        "num pages": num_docs,
        "num words": num_words,
        "num chars": num_chars,
    }
    if folder:
        json.dump(
            stats,
            open(stat_filename, "w", encoding="utf8"),
            indent=2,
            ensure_ascii=False,
        )
    return stats


def simple_slugify(name):
    return re.sub(r"[ :/]", "--", name).strip("_-")


if __name__ == "__main__":
    main()
