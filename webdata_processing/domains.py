import argparse

from datatrove.data import Document
from datatrove.executor import SlurmPipelineExecutor
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import ParquetWriter
from datatrove.pipeline.writers.disk_base import DiskWriter


class CanFetchFilter(BaseFilter):
    name = "👮 Can Fetch URLs"

    def __init__(
        self,
        exclusion_writer: DiskWriter = None,
    ):
        super().__init__(exclusion_writer)
        self._valid_domains = None

    @property
    def valid_domains(self):
        import json

        if self._valid_domains is None:
            file_path = "/gpfsscratch/rech/qgz/uzq54wg/valid_domains_redpajama_16M.json"
            with open(file_path) as fp:
                self._valid_domains = set(json.load(fp))
        return self._valid_domains

    def filter(self, doc: Document) -> bool | tuple[bool, str]:  # noqa # C901
        if doc.metadata["url"] in self.valid_domains:
            return True
        else:
            return False, "👮: Cannot fetch this domain"


def get_args():
    parser = argparse.ArgumentParser(description="Process some configurations.")

    # Adding arguments
    parser.add_argument(
        "--dump-to-process", type=str, default="2023-14", help="Specify the dump to process. Default is '2023-14'."
    )
    parser.add_argument("--language", type=str, default="fr", help="Specify the language. Default is 'fr'.")
    parser.add_argument(
        "--main-output-path",
        type=str,
        default="/gpfsscratch/rech/qgz/uzq54wg/processed_redpajama",
        help="Specify the main output path. Default is '/gpfsscratch/rech/qgz/uzq54wg/processed_redpajama'.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    DUMP_TO_PROCESS = args.dump_to_process
    LANGUAGE = args.language
    MAIN_OUTPUT_PATH = args.main_output_path
    FILTERING_OUTPUT_PATH = f"{MAIN_OUTPUT_PATH}/base_processing"
    URL_FILTERING_OUTPUT_PATH = f"{MAIN_OUTPUT_PATH}/url_filtering_16M"

    main_processing_executor = SlurmPipelineExecutor(
        job_name=f"{DUMP_TO_PROCESS}--{LANGUAGE}",
        pipeline=[
            ParquetReader(f"{FILTERING_OUTPUT_PATH}/output/{LANGUAGE}/{DUMP_TO_PROCESS}"),
            CanFetchFilter(),
            ParquetWriter(f"{URL_FILTERING_OUTPUT_PATH}/output/{LANGUAGE}/{DUMP_TO_PROCESS}"),
        ],
        sbatch_args={"account": "qgz@cpu"},
        tasks=10,
        cpus_per_task=2,
        time="1:00:00",
        logging_dir=f"{MAIN_OUTPUT_PATH}/logs/url_filtering_16M/{LANGUAGE}/{DUMP_TO_PROCESS}",
        slurm_logs_folder=f"logs/url_filtering_16M/{LANGUAGE}/{DUMP_TO_PROCESS}",  # must be local
        randomize_start_duration=180,  # don't hit the bucket all at once with the list requests
        qos="qos_cpu-t3",
        partition="cpu_p1",
        condaenv="/gpfsscratch/rech/qgz/uzq54wg/datatrove",
    )

    main_processing_executor.run()
