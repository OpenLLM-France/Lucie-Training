# Copyright 2023 Together Computer
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""RedPajama V2: Quality annotated Web Text Documents."""

import gzip
import json
import traceback

import datasets
import pyarrow.parquet as pq

logger = datasets.logging.get_logger(__name__)

_DESCRIPTION = """\
RedPajama V2: an Open Dataset for Training Large Language Models
"""

_URL_BASE = "/gpfsdswork/dataset/RedPajama-V2/v1.0.0"  # "https://data.together.xyz/redpajama-data-v2/v1.0.0" #
_LANGUAGES = ("en", "de", "fr", "es", "it")
_MISSING_FILES_PATTERN = "urls/missing-{component}.txt"
_NUM_SHARDS = 5000
_SUBSAMPLE_FILE_COUNTS = {"sample-10B": 1, "sample-100B": 10, "sample-1T": 100}

_CC_SNAPSHOT_IDS = (
    "2014-15",
    "2014-23",
    "2014-35",
    "2014-41",
    "2014-42",
    "2014-49",
    "2014-52",
    "2015-14",
    "2015-22",
    "2015-27",
    "2015-32",
    "2015-35",
    "2015-40",
    "2015-48",
    "2016-07",
    "2016-18",
    "2016-22",
    "2016-26",
    "2016-30",
    "2016-36",
    "2016-40",
    "2016-44",
    "2016-50",
    "2017-04",
    "2017-09",
    "2017-17",
    "2017-22",
    "2017-26",
    "2017-30",
    "2017-34",
    "2017-39",
    "2017-43",
    "2017-47",
    "2017-51",
    "2018-05",
    "2018-09",
    "2018-13",
    "2018-17",
    "2018-22",
    "2018-26",
    "2018-30",
    "2018-34",
    "2018-39",
    "2018-43",
    "2018-47",
    "2018-51",
    "2019-04",
    "2019-09",
    "2019-13",
    "2019-18",
    "2019-22",
    "2019-26",
    "2019-30",
    "2019-35",
    "2019-39",
    "2019-43",
    "2019-47",
    "2019-51",
    "2020-05",
    "2020-10",
    "2020-16",
    "2020-24",
    "2020-29",
    "2020-34",
    "2020-40",
    "2020-45",
    "2020-50",
    "2021-04",
    "2021-10",
    "2021-17",
    "2021-21",
    "2021-25",
    "2021-31",
    "2021-39",
    "2021-43",
    "2021-49",
    "2022-05",
    "2022-21",
    "2022-27",
    "2022-33",
    "2022-40",
    "2022-49",
    "2023-06",
    "2023-14",
)


class RedPajamaDataV2Config(datasets.BuilderConfig):
    """BuilderConfig for RedPajama."""

    def __init__(self, *args, **kwargs):
        """BuilderConfig for RedPajama.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(RedPajamaDataV2Config, self).__init__(**kwargs)  # noqa # C401
        self.partition: str = kwargs.pop("partition", "all")
        self.snapshots: list[str] = kwargs.pop("snapshots", _CC_SNAPSHOT_IDS)
        self.languages: list[str] = kwargs.pop("languages", _LANGUAGES)


class RedPajamaV2(datasets.GeneratorBasedBuilder):
    """RedPajama V2: Quality annotated Web Text Documents."""

    BUILDER_CONFIGS = [
        RedPajamaDataV2Config(
            name="sample",
            version=datasets.Version("1.0.0", ""),
            description="RedPajamaV2 Sample",
        ),
        RedPajamaDataV2Config(
            name="sample-10B",
            version=datasets.Version("1.0.0", ""),
            description="RedPajamaV2 Sample with 10B tokens",
        ),
        RedPajamaDataV2Config(
            name="sample-100B",
            version=datasets.Version("1.0.0", ""),
            description="RedPajamaV2 Sample with 100B tokens",
        ),
        RedPajamaDataV2Config(
            name="sample-1T",
            version=datasets.Version("1.0.0", ""),
            description="RedPajamaV2 Sample with 1T tokens",
        ),
        RedPajamaDataV2Config(
            name="default",
            version=datasets.Version("1.0.0", ""),
            description="RedPajamaV2",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "raw_content": datasets.Value("string"),
                    "doc_id": datasets.Value("string"),
                    "meta": datasets.Value("string"),
                    "quality_signals": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators_sample(self, dl_manager):
        # fetch list of base tags
        sample_base_tags_fp = dl_manager.download_and_extract("sample/sample_listings.txt")
        with open(sample_base_tags_fp) as fd:
            sample_base_tags = [line.strip() for line in fd]

        # fetch documents
        logger.info(f"Downloading {len(sample_base_tags)} documents files.")
        documents_files = dl_manager.download(
            {base_tag: f"sample/documents/{base_tag}.json.gz" for base_tag in sample_base_tags}
        )

        # fetch quality signals
        logger.info(f"Downloading {len(sample_base_tags)} quality signals files.")
        quality_signals_files = dl_manager.download(
            {base_tag: f"sample/quality_signals/{base_tag}.signals.json.gz" for base_tag in sample_base_tags}
        )

        # fetch ids of duplicates
        logger.info(f"Downloading {len(sample_base_tags)} duplicates ids files.")
        duplicates_ids_files = dl_manager.download(
            {base_tag: f"sample/duplicates/{base_tag}.duplicates.parquet" for base_tag in sample_base_tags}
        )

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "base_tags": sample_base_tags,
                    "documents_files": documents_files,
                    "quality_signals_files": quality_signals_files,
                    "duplicates_ids_files": duplicates_ids_files,
                },
            )
        ]

    def _split_generators_full(self, dl_manager):  # noqa # C901
        snapshots = getattr(self.config, "snapshots", _CC_SNAPSHOT_IDS)
        languages = getattr(self.config, "languages", _LANGUAGES)
        partition = getattr(self.config, "partition", "all")

        if self.config.name in ("sample-10B", "sample-100B", "sample-1T"):
            partition = "head_middle"
            languages = _LANGUAGES
            snapshots = _CC_SNAPSHOT_IDS
            num_shards = _SUBSAMPLE_FILE_COUNTS[self.config.name]
        else:
            num_shards = _NUM_SHARDS

        if partition == "all":
            partitions = ["head", "middle", "tail"]
        elif partition == "head_middle":
            partitions = ["head", "middle"]
        elif partition == "tail":
            partitions = [partition]
        else:
            raise ValueError(f"invalid partition: {partition}")

        # fetch list of missing files (e.g., missing duplicates or corrupted documents and
        # quality signal files)
        missing_files_paths = dl_manager.download_and_extract(
            {
                component: _URL_BASE + _MISSING_FILES_PATTERN.format(component=component)
                for component in ("documents", "signals", "duplicates")
            }
        )

        missing_files = {}
        for component, missing_file in missing_files_paths.items():
            with open(missing_file, encoding="utf-8") as f:
                missing_files[component] = set(line.strip() for line in f)  # noqa # C401

        # build list of urls to fetch
        documents_urls = {}
        quality_signals_urls = {}
        duplicates_ids_urls = {}
        base_tags = []

        for lang in languages:
            for snapshot in snapshots:
                for part in partitions:
                    for n in range(num_shards):
                        base_tag = f"{snapshot}/{n:04d}/{lang}_{part}"
                        base_tags.append(base_tag)

                        # documents
                        url = f"{_URL_BASE}/documents/{base_tag}.json.gz"
                        if url not in missing_files["documents"]:
                            documents_urls[base_tag] = url

                        # quality signals
                        url = f"{_URL_BASE}/quality_signals/{base_tag}.signals.json.gz"
                        if url not in missing_files["signals"]:
                            quality_signals_urls[base_tag] = url

                        # duplicates
                        url = f"{_URL_BASE}/duplicates/{base_tag}.duplicates.parquet"
                        if url not in missing_files["duplicates"]:
                            duplicates_ids_urls[base_tag] = url

        # download documents files
        logger.info(f"Downloading {len(documents_urls)} documents files.")
        documents_files = dl_manager.download(documents_urls)

        # download quality signals files
        logger.info(f"Downloading {len(quality_signals_urls)} quality signals files.")
        quality_signals_files = dl_manager.download(quality_signals_urls)

        # download duplicates ids files
        logger.info(f"Downloading {len(duplicates_ids_urls)} duplicates ids files.")
        duplicates_ids_files = dl_manager.download(duplicates_ids_urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "base_tags": base_tags,
                    "documents_files": documents_files,
                    "quality_signals_files": quality_signals_files,
                    "duplicates_ids_files": duplicates_ids_files,
                },
            )
        ]

    def _split_generators(self, dl_manager):
        if self.config.name == "sample":
            return self._split_generators_sample(dl_manager)

        return self._split_generators_full(dl_manager)

    def _generate_examples(self, base_tags, documents_files, quality_signals_files, duplicates_ids_files):
        key = 0
        for base_tag in base_tags:
            doc_file = documents_files.get(base_tag)
            qs_file = quality_signals_files.get(base_tag)
            dupe_file = duplicates_ids_files.get(base_tag)

            if doc_file is None:
                continue

            for sample in self.__get_generator(base_tag, doc_file, qs_file, dupe_file):
                yield key, sample
                key += 1

    def __get_generator(self, base_tag, doc_file, qs_file, dupe_file):
        if "_tail" in base_tag:
            yield from self._handle_tail(base_tag, doc_file, qs_file, dupe_file)
        else:
            yield from self._handle_head_middle(base_tag, doc_file, qs_file, dupe_file)

    def _handle_tail(self, base_tag, doc_file, qs_file, dupe_file):
        try:
            with gzip.open(doc_file, "rt", encoding="utf-8") as df:
                for row, doc in enumerate(df):
                    doc_id = f"{base_tag}.json.gz/{row}"
                    try:
                        yield self.handle_record("tail", doc_id, doc, None, None)
                    except Exception:
                        logger.warning(f"failed handling row {row} in {doc_file}")
                        traceback.print_exc()
                        continue

        except gzip.BadGzipFile:
            # skip broken gzip files
            print(f"BadGzipFile: {doc_file, qs_file}")
            traceback.print_exc()
            return

    def _handle_head_middle(self, base_tag, doc_file, qs_file, dupe_file):
        if qs_file is None:
            yield from self._handle_tail(base_tag, doc_file, None, None)
            return

        # load duplicates
        try:
            with open(dupe_file, "rb") as df:
                duplicates = set(pq.read_table(df, columns=["doc_id"], use_pandas_metadata=False)["doc_id"].to_pylist())
        except Exception:
            logger.warning(f"no duplicate ids found for {base_tag}")
            duplicates = set()

        try:
            with gzip.open(doc_file, "rt", encoding="utf-8") as df:
                with gzip.open(qs_file, "rt", encoding="utf-8") as qf:
                    for row, (doc, qs) in enumerate(zip(df, qf)):
                        doc_id = f"{base_tag}.json.gz/{row}"

                        try:
                            yield self.handle_record(
                                part="head_middle",
                                doc_id=doc_id,
                                doc=doc,
                                qs=qs,
                                is_duplicate=doc_id in duplicates,
                            )
                        except Exception:
                            logger.warning(f"failed handling row {row} in {doc_file} ({qs_file})")
                            traceback.print_exc()
                            continue

        except gzip.BadGzipFile:
            # skip broken gzip files
            print(f"BadGzipFile: {doc_file, qs_file}")
            traceback.print_exc()
            return

    @staticmethod
    def handle_record(part, doc_id, doc, qs, is_duplicate=None):
        doc = json.loads(doc)
        qs = json.loads(qs) if qs is not None else {}

        meta = {
            "url": doc["url"],
            "partition": part,
            "language": doc["language"],
            "source_domain": doc["source_domain"],
            "date_download": doc["date_download"],
            "digest": doc["digest"],
        }

        quality_signals = qs.get("quality_signals", {})
        quality_signals["is_duplicate"] = is_duplicate

        return {
            "raw_content": doc["raw_content"],
            "doc_id": doc_id,
            "meta": json.dumps(meta),
            "quality_signals": json.dumps(quality_signals),
        }
