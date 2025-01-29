import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import regex as re

_parent_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_csv_weights_folder = os.path.join(_parent_folder, "assets")


def load_df(path):
    df = pd.read_csv(path)
    df = df[["prefix", "reweighted_count", "new_ratio"]]
    df["name"] = df["prefix"].apply(lambda x: x.split("/")[-1])
    return df


if __name__ == "__main__":
    default_path = "/data-storage/storage0"
    for path in [
        "/data-storage/storage0",
        "/lustre/fsn1/projects/rech/qgz/commun/preprocessed_data/Lucie",
    ]:
        if os.path.exists(path):
            default_path = path
            break

    parser = argparse.ArgumentParser(
        description="Prints a string with all tokenized data files (prefixes) and their respective weights.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "folder",
        type=str,
        help="Path to weights csv",
        default=_csv_weights_folder,
        nargs="?",
    )
    parser.add_argument(
        "--start_path",
        type=str,
        help="Path to tokenized data",
        default=default_path,
        nargs="?",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="To print debug output",
    )
    parser.add_argument(
        "--plot-pie",
        action="store_true",
        default=False,
        help="To plot a pie chart with distribution of each dataset (in amount of tokens)",
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

    if args.verbose:
        print(f"Proportion of docs longer than 4k: {df_long['new_ratio'].sum()}")

    cat_df = pd.concat([df_long, df_short]).dropna()
    cat_df = cat_df.sort_values("new_ratio", ascending=False)

    # Remove datasets with number of estimated tokens lower than 32k!
    cat_df["token_estimation"] = cat_df["new_ratio"].apply(lambda x: x * 5e9)
    if args.verbose:
        print("\n Datasets with low nnumber of tokens")
        print(cat_df[cat_df["token_estimation"] < 32000][["prefix", "token_estimation"]])
    cat_df = cat_df[cat_df["token_estimation"] > 32000]

    num_colors = 20
    # chosen_colors = [plt.cm.rainbow(i / num_colors) for i in range(num_colors)]
    chosen_colors = [plt.cm.tab20(i / num_colors) for i in range(num_colors)]
    prefix_to_color = {}

    ratios = {}
    colors = {}

    def norm_name(f):
        short = "length-0-4096" in f
        f = f.split("/")[-1]
        f = f.replace("_text_document", "")
        return f + (" (short)" if short else " (long)")

    def color(f):
        f = norm_name(f)
        short = f.endswith(" (short)")
        f = f.split(" (")[0]
        if f not in prefix_to_color:
            prefix_to_color[f] = len(prefix_to_color)
        color = chosen_colors[prefix_to_color[f] % len(chosen_colors)]
        if short:
            assert len(color) == 4
            color = list(color)
            color[-1] = 0.5
            color = tuple(color)
        return color

    for _, row in cat_df.iterrows():
        prefix = os.path.join(args.start_path, row["prefix"])
        new_ratio = row["new_ratio"]
        # if 'Claire--fr--ESLO_text_document' in prefix:
        #     if args.verbose:
        #         print('\n\nSMALL')
        #         print(new_ratio)
        #     break
        ratios[norm_name(prefix)] = float(new_ratio)
        colors[norm_name(prefix)] = color(prefix)
        # Print the weight (expected output)
        sweight = f"{new_ratio:11.9f}"
        # Check that nothing was rounded to weight=0
        if not re.search(r"[^\.0]", sweight):
            pass
        else:
            print(f"{sweight} {prefix} ", end="")

    if args.plot_pie:
        plt.pie(ratios.values(), labels=ratios.keys(), autopct="%1.1f%%", colors=[colors[k] for k in ratios.keys()])
        plt.show()
