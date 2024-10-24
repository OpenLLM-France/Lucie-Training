import argparse

import regex as re
from datatrove.executor import SlurmPipelineExecutor
from datatrove.pipeline.dedup import MinhashDedupCluster, MinhashDedupFilter, MinhashDedupSignature
from datatrove.pipeline.dedup.minhash import MinhashConfig, MinhashDedupBuckets
from datatrove.pipeline.formatters.base import BaseFormatter
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import ParquetWriter


class CorrectPII(BaseFormatter):
    name = "🤒 Correct PII"

    def __init__(
        self,
    ):
        super().__init__()

    def format(self, text: str) -> str:
        import random

        list_ips = [
            "22.214.171.124",
            "126.96.36.199",
            "188.8.131.52",
            "184.108.40.206",
            "220.127.116.11",
            "18.104.22.168",
        ]
        list_emails = ["email@example.com", "firstname.lastname@example.org"]
        text = re.sub("<email>", random.choice(list_emails), text)
        text = re.sub("<ip>", random.choice(list_ips), text)
        return text


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
        default="/lustre/fsn1/projects/rech/qgz/uzq54wg/processed_redpajama",
        help="Specify the main output path. Default is '/lustre/fsn1/projects/rech/qgz/uzq54wg/processed_redpajama'.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    DUMP_TO_PROCESS = args.dump_to_process
    LANGUAGE = args.language
    MAIN_OUTPUT_PATH = args.main_output_path
    FILTERING_OUTPUT_PATH = f"{MAIN_OUTPUT_PATH}/base_processing"

    minhash_config = MinhashConfig(
        num_buckets=14,
        hashes_per_bucket=8,
        n_grams=5,
    )

    MINHASH_BASE_PATH = f"{MAIN_OUTPUT_PATH}/minhash"

    LOGS_FOLDER = f"{MAIN_OUTPUT_PATH}/logs/minhash"
    LOCAL_LOGS_FOLDER = "logs/minhash"

    TOTAL_TASKS = 50

    # this is the original data that we want to deduplicate
    INPUT_READER = ParquetReader(
        f"{FILTERING_OUTPUT_PATH}/output/{LANGUAGE}/{DUMP_TO_PROCESS}"
    )  # this is the output from the first part

    # stage 1 computes minhash signatures for each task (each task gets a set of files)
    stage1 = SlurmPipelineExecutor(
        job_name=f"mh1_{DUMP_TO_PROCESS}--{LANGUAGE}",
        pipeline=[
            INPUT_READER,
            MinhashDedupSignature(
                output_folder=f"{MINHASH_BASE_PATH}/{LANGUAGE}/{DUMP_TO_PROCESS}/signatures", config=minhash_config
            ),
        ],
        sbatch_args={"account": "qgz@cpu"},
        tasks=TOTAL_TASKS,
        time="5:00:00",
        qos="qos_cpu-t3",
        partition="cpu_p1",
        condaenv="/lustre/fsn1/projects/rech/qgz/uzq54wg/envs/datatrove",
        logging_dir=f"{LOGS_FOLDER}/signatures/{LANGUAGE}/{DUMP_TO_PROCESS}",
        slurm_logs_folder=f"{LOCAL_LOGS_FOLDER}/signatures/{LANGUAGE}/{DUMP_TO_PROCESS}",
        randomize_start_duration=180,
    )

    stage2 = SlurmPipelineExecutor(
        job_name=f"mh2_{DUMP_TO_PROCESS}--{LANGUAGE}",
        pipeline=[
            MinhashDedupBuckets(
                input_folder=f"{MINHASH_BASE_PATH}/{LANGUAGE}/{DUMP_TO_PROCESS}/signatures",
                output_folder=f"{MINHASH_BASE_PATH}/{LANGUAGE}/{DUMP_TO_PROCESS}/buckets",
                config=minhash_config,
            ),
        ],
        sbatch_args={"account": "qgz@cpu"},
        tasks=minhash_config.num_buckets * 2,  # the code supports parallelizing each bucket.
        randomize_start_duration=180,
        logging_dir=f"{LOGS_FOLDER}/buckets/{LANGUAGE}/{DUMP_TO_PROCESS}",
        slurm_logs_folder=f"{LOCAL_LOGS_FOLDER}/buckets/{LANGUAGE}/{DUMP_TO_PROCESS}",
        qos="qos_cpu-t3",
        partition="cpu_p1",
        condaenv="/lustre/fsn1/projects/rech/qgz/uzq54wg/envs/datatrove",
        time="02:00:00",
        cpus_per_task=1,  # you can add run more (smaller) tasks if you do not have a lot of memory
        depends=stage1,
    )

    stage3 = SlurmPipelineExecutor(
        job_name=f"mh3_{DUMP_TO_PROCESS}--{LANGUAGE}",
        pipeline=[
            MinhashDedupCluster(
                input_folder=f"{MINHASH_BASE_PATH}/{LANGUAGE}/{DUMP_TO_PROCESS}/buckets",
                output_folder=f"{MINHASH_BASE_PATH}/{LANGUAGE}/{DUMP_TO_PROCESS}/remove_ids",
                config=minhash_config,
            ),
        ],
        sbatch_args={"account": "qgz@cpu"},
        tasks=1,  # this step runs on a single task
        logging_dir=f"{LOGS_FOLDER}/clustering/{LANGUAGE}/{DUMP_TO_PROCESS}",
        slurm_logs_folder=f"{LOCAL_LOGS_FOLDER}/clustering/{LANGUAGE}/{DUMP_TO_PROCESS}",
        qos="qos_cpu-t3",
        partition="cpu_p1",
        condaenv="/lustre/fsn1/projects/rech/qgz/uzq54wg/envs/datatrove",
        time="20:00:00",  # and can also be quite slow. Usually not this slow though
        cpus_per_task=8,  # if you dedup a full dump, you do need a lot of memory for this one
        depends=stage2,
    )

    stage4 = SlurmPipelineExecutor(
        job_name=f"mh4_{DUMP_TO_PROCESS}--{LANGUAGE}",
        pipeline=[
            INPUT_READER,
            MinhashDedupFilter(input_folder=f"{MINHASH_BASE_PATH}/{LANGUAGE}/{DUMP_TO_PROCESS}/remove_ids"),
            CorrectPII(),
            ParquetWriter(f"{MINHASH_BASE_PATH}/{LANGUAGE}/{DUMP_TO_PROCESS}/deduped_output"),
        ],
        sbatch_args={"account": "qgz@cpu"},
        tasks=TOTAL_TASKS,
        logging_dir=f"{LOGS_FOLDER}/filtering_v2/{LANGUAGE}/{DUMP_TO_PROCESS}",
        slurm_logs_folder=f"{LOCAL_LOGS_FOLDER}/filtering_v2/{LANGUAGE}/{DUMP_TO_PROCESS}",
        qos="qos_cpu-t3",
        partition="cpu_p1",
        cpus_per_task=2,
        randomize_start_duration=180,  # don't hit the bucket all at once with the list requests
        condaenv="/lustre/fsn1/projects/rech/qgz/uzq54wg/envs/datatrove",
        time="1:00:00",
    )

    # launch dedup pipelines
    stage4.run()
