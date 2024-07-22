import argparse

from datatrove.executor import SlurmPipelineExecutor
from datatrove.pipeline.dedup import MinhashDedupCluster, MinhashDedupFilter, MinhashDedupSignature
from datatrove.pipeline.dedup.minhash import MinhashConfig, MinhashDedupBuckets
from datatrove.pipeline.formatters import PIIFormatter
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import ParquetWriter


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

    minhash_config = MinhashConfig(
        num_buckets=14,
        hashes_per_bucket=8,
        n_grams=5,
    )

    MINHASH_BASE_PATH = f"{MAIN_OUTPUT_PATH}/minhash"

    LOGS_FOLDER = f"{MAIN_OUTPUT_PATH}/logs/minhash"
    LOCAL_LOGS_FOLDER = "logs/minhash"

    TOTAL_TASKS = 10

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
        time="2:00:00",
        qos="qos_cpu-t3",
        partition="cpu_p1",
        condaenv="/gpfsscratch/rech/qgz/uzq54wg/datatrove",
        logging_dir=f"{LOGS_FOLDER}/signatures",
        slurm_logs_folder=f"{LOCAL_LOGS_FOLDER}/signatures/slurm_logs",
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
        tasks=1,  # the code supports parallelizing each bucket. here we run 1
        randomize_start_duration=180,
        logging_dir=f"{LOGS_FOLDER}/buckets",
        slurm_logs_folder=f"{LOCAL_LOGS_FOLDER}/buckets/slurm_logs",
        qos="qos_cpu-t3",
        partition="cpu_p1",
        condaenv="/gpfsscratch/rech/qgz/uzq54wg/datatrove",
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
        logging_dir=f"{LOGS_FOLDER}/clustering",
        slurm_logs_folder=f"{LOCAL_LOGS_FOLDER}/clustering/slurm_logs",
        qos="qos_cpu-t3",
        partition="cpu_p1",
        condaenv="/gpfsscratch/rech/qgz/uzq54wg/datatrove",
        time="5:00:00",  # and can also be quite slow. Usually not this slow though
        cpus_per_task=8,  # if you dedup a full dump, you do need a lot of memory for this one
        depends=stage2,
    )

    # stage3.run()

    stage4 = SlurmPipelineExecutor(
        job_name=f"mh4_{DUMP_TO_PROCESS}--{LANGUAGE}",
        pipeline=[
            INPUT_READER,
            MinhashDedupFilter(input_folder=f"{MINHASH_BASE_PATH}/{LANGUAGE}/{DUMP_TO_PROCESS}/remove_ids"),
            PIIFormatter(),
            ParquetWriter(f"{MINHASH_BASE_PATH}/{LANGUAGE}/{DUMP_TO_PROCESS}/deduped_output"),
        ],
        sbatch_args={"account": "qgz@cpu"},
        tasks=TOTAL_TASKS,
        logging_dir=f"{LOGS_FOLDER}/filtering",
        slurm_logs_folder=f"{LOCAL_LOGS_FOLDER}/filtering/slurm_logs",
        qos="qos_cpu-t3",
        partition="cpu_p1",
        condaenv="/gpfsscratch/rech/qgz/uzq54wg/datatrove",
        time="2:00:00",
        depends=stage3,
    )

    # launch dedup pipelines
    stage4.run()
