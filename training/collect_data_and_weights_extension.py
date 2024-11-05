import argparse
import os

import pandas as pd
import regex as re

_parent_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_csv_weights_folder = os.path.join(_parent_folder, "chronicles", "pretrain")


def load_df(path):
    df = pd.read_csv(path)
    df = df[["prefix", "reweighted_count", "new_ratio"]]
    df["name"] = df["prefix"].apply(lambda x: x.split("/")[-1])
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prints a string with all tokenized data files (prefixes) and their respective weights.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "folder",
        type=str,
        help="Path to tokenized data",
        default=os.path.join(_csv_weights_folder, "weights_output"),
        nargs="?",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="To print debug output",
    )
    args = parser.parse_args()

    ### Load main and process Web data
    df_main = load_df(os.path.join(args.folder, "training_weights.csv"))
    if args.verbose:
        print(df_main)

    def define_group(name):
        if "FineWebEdu" in name:
            return "FineWebEdu--cc-main-2024_text_document"
        if "RedPajama--fr" in name:
            return "RedPajama--fr--2023_text_document"
        if "RedPajama--de" in name:
            return "RedPajama--de--2023_text_document"
        if "RedPajama--es" in name:
            return "RedPajama--es--2023_text_document"
        if "RedPajama--it" in name:
            return "RedPajama--it--2023_text_document"
        return name

    df_main["name"] = df_main["name"].apply(define_group)
    df_main_pro = df_main.groupby("name").sum().reset_index()
    if args.verbose:
        print(f"Main length: {len(df_main_pro)}")

    ### load short and long context
    df_long = load_df(os.path.join(args.folder, "training_weights_long.csv"))
    if args.verbose:
        print(f"Long length: {len(df_long)}")

    df_short = load_df(os.path.join(args.folder, "training_weights_short.csv"))
    if args.verbose:
        print(f"Short length: {len(df_short)}")

    df_long_short = df_long.merge(df_short, on="name", how="outer", suffixes=("_long", "_short"))
    df_long_short["reweighted_count_long"] = df_long_short["reweighted_count_long"].fillna(0)
    df_long_short["reweighted_count_short"] = df_long_short["reweighted_count_short"].fillna(0)
    if args.verbose:
        print(f"Short+Long length: {len(df_long_short)}")

    ### Merge short+long+main
    df = df_main_pro.merge(df_long_short, how="outer", on="name")

    ### Compare to check
    wrong_sum = df[
        df.apply(lambda x: x["reweighted_count"] != (x["reweighted_count_long"] + x["reweighted_count_short"]), axis=1)
    ]["name"]
    if args.verbose:
        print(f"\nDatasets with wrond sum (should be web data only):\n{wrong_sum}")

    ### Calculating the new weights!
    df["prop_short"] = df["reweighted_count_short"] / (df["reweighted_count_short"] + 10 * df["reweighted_count_long"])
    df["prop_long"] = 1 - df["prop_short"]
    df = df.drop(["reweighted_count_long", "reweighted_count_short"], axis=1)

    df["new_ratio_long"] = df["new_ratio"] * df["prop_long"]
    df["new_ratio_short"] = df["new_ratio"] * df["prop_short"]

    df_long = df[["prefix_long", "new_ratio_long"]]
    df_long.columns = ["prefix", "new_ratio"]
    df_short = df[["prefix_short", "new_ratio_short"]]
    df_short.columns = ["prefix", "new_ratio"]

    cat_df = pd.concat([df_long, df_short]).dropna()
    cat_df = cat_df.sort_values("new_ratio")

    if args.verbose:
        print(f"Prop docs longer than 4k: {df_long['new_ratio'].sum()}")

    for _, row in cat_df.iterrows():
        prefix = row["prefix"]
        new_ratio = row["new_ratio"]
        # Print the weight (expected output)
        sweight = f"{new_ratio:11.9f}"
        # Check that nothing was rounded to weight=0
        if not re.search(r"[^\.0]", sweight):
            pass
        else:
            print(f"{sweight} {prefix} ", end="")
