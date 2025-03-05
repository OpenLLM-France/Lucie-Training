import argparse
import json
import os

import pandas as pd


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(args)

    for type_file in ["can_fetch", "logs"]:  # , 'robots_txt_files'
        data_dir = os.path.join(args.output_path, type_file)
        dfs = []
        for file in os.listdir(data_dir):
            df = pd.read_json(os.path.join(data_dir, file), lines=True).drop_duplicates()
            dfs.append(df)

        df_out = pd.concat(dfs)
        df_out.to_csv(os.path.join(args.output_path, f"{type_file}.csv"))

    # From can fetch list the valid ones
    df = pd.read_csv(os.path.join(args.output_path, "can_fetch.csv"))
    df = df[df["can_fetch"]]

    valid_domains = list(df["domain"])

    with open(os.path.join(args.output_path, "valid_domains.json"), "w") as fp:
        json.dump(valid_domains, fp)
