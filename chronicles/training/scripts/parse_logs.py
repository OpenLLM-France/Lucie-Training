"""Parse Megatron Deepspeed log file and and get a structured CSV."""
import re
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd

def parse_logs_file(input_folder: str, output_folder) -> None:
    df = []
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)
    for zero in Path(input_folder).glob("*/"):
        for batch in zero.glob("*/"):
            for log_file in batch.glob("iterations/*.iterations"):
                if "tp4" in log_file.stem:
                    print(log_file)
                    continue
                nnodes = re.search("\d+nodes", log_file.name).group(0)
                nnodes = re.search("\d+", nnodes).group(0)
                tp = re.search("tp\d+", log_file.name).group(0)
                pp = re.search("(pp\d+|no_pp)", log_file.name).group(0)
                pp = pp.replace("no_pp", "pp1")
                ngpus = 8 * int(nnodes)
                with open(log_file, "r", encoding='utf-8', errors="ignore") as logs:
                    for idx, line in enumerate(logs, 1):
                        # skip the first iteration
                        if idx == 1:
                            continue
                        # only consider the 50 first iteration
                        if idx == 50:
                            break
                        # only take the first 50 iterations and not the first as the iteration time is low for the first iteration and the first iteration after a validation step.
                        # TODO: enumerate and ignore the first and the step after eval
                        _, consumed_samples, consumed_tokens, iter_time, _, _, loss, *_, samples_per_seconds, tflops, _ = line.split("|")
                        # print(consumed_samples, consumed_tokens, iter_time, loss, samples_per_seconds, tflops)
                        consumed_samples = re.search(r'\d+', consumed_samples).group(0)
                        consumed_tokens = re.search(r'\d+', consumed_tokens).group(0)
                        iter_time = re.search(r'\d+.\d+', iter_time).group(0)
                        loss = re.search(r'\d+.\d+E\+\d+', loss).group(0)
                        samples_per_seconds = re.search(r'\d+.\d+', samples_per_seconds).group(0)
                        tflops = re.search(r'\d+.\d+', tflops).group(0)
                        df.append(
                            {
                                "consumed_samples": consumed_samples,
                                "consumed_tokens": consumed_tokens,
                                "iter_time": iter_time,
                                "loss": loss,
                                "samples_per_seconds": samples_per_seconds,
                                "TFLOPs": tflops,
                                "ngpus": ngpus,
                                "strategy": f"{tp}-{pp}",
                                "ntokens_per_second" : float(samples_per_seconds) * 2048,
                                "batch": batch.stem,
                                "stage": zero.stem
                            }
                        )
    df = pd.DataFrame(df)
    output_filename = "all"
    df.to_csv(output_folder / f"{output_filename}.csv", index=False)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_folder",
                        type=str,
                        help="The input log folder containing log files from Megatron Deepspeed",
                        required=True)
    parser.add_argument("-o", "--output_folder",
                        type=str,
                        help="Where to store the output CSV file.",
                        required=True)
    args = parser.parse_args()
    parse_logs_file(args.input_folder, args.output_folder)