import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PLOT_MORE_THAN_32k = True
PLOT_BEST = True
PLOT_GALLICA_PERSEE = False
PLOT_LUCIE_32k = False
SHOW_TOKENIZER_SIZE = True
SEPARATE_LAST_DATASETS = 1  # Code
TRANSPARENT_BACKGROUND = False
FONT_SIZE = 16

SYSTEMS = [
    # "gpt2",
    "google/gemma-7b",
    "bigscience/bloom-7b1",
    "GPT-4",
    "tiiuae/falcon-7b",
    "meta-llama/Llama-2-7b-hf",
    "mistralai/Mistral-7B-v0.1",
    "croissantllm/CroissantLLMBase",
]

NUM_BASELINES_ = 7
NUM_MORE_THAN_32K_ = 4
NUM_MORE_THAN_32K = None  # NUM_MORE_THAN_32K_

COLORS = [
    "dodgerblue",
    "yellow",
    "black",
    "mediumblue",  # "navy",
    "goldenrod",
    "darkorange",
    "blue",
]
if False:
    ALPHAS = [0.5] * NUM_MORE_THAN_32K_
    LINESTYLE = ["-"] * len(COLORS)
else:
    ALPHAS = []
    LINESTYLE = [":"] * NUM_MORE_THAN_32K_ + ["-"] * (len(COLORS) - NUM_MORE_THAN_32K_)

LINEWIDTH = [4] * NUM_BASELINES_  # [4] * 3 + [2] * (len(COLORS)-3)
DEFAULT_LINEWIDTH = 3

if not PLOT_MORE_THAN_32k:
    NUM_BASELINES = 3
    SYSTEMS = SYSTEMS[4:]
    COLORS = COLORS[4:]
    LINESTYLE = LINESTYLE[4:]
    LINEWIDTH = LINEWIDTH[4:]


# LINEWIDTH+= [8]

COLORS += [
    "darkviolet",
    "r",
    "r",
    "darkviolet",
    "darkviolet",
    "violet",
    "violet",
    "magenta",  # "magenta",
    "purple",  # "purple",
    "black",
]
SYSTEMS += [
    # "Lucie-v2.4-space_prefix_all",
    "Lucie2.9",
]

LINESTYLE += [
    "-",
    "-",
    "--",
    "-",
    "--",  # ':',
    "-",
    "--",  # '--',
    "-",  # '--',
    "--",
]

if PLOT_MORE_THAN_32k:
    # SYSTEMS = SYSTEMS[:4] + ["Lucie2.9_65k", "Lucie2.10_65k", "Lucie2.10_65kb"] + SYSTEMS[4:]
    # COLORS = COLORS[:4] + ["r", "m", "violet"] + COLORS[4:]
    # LINESTYLE = LINESTYLE[:4] + ["--", ":", ":"] + LINESTYLE[4:]
    # LINEWIDTH = LINEWIDTH[:4] + [DEFAULT_LINEWIDTH, DEFAULT_LINEWIDTH, DEFAULT_LINEWIDTH] + LINEWIDTH[4:]

    # SYSTEMS = SYSTEMS[:4] + ["Lucie2.10_65k", "Lucie2.10_65k_orig", "Lucie2.11_65k"] + SYSTEMS[4:]
    # COLORS = COLORS[:4] + ["r", "m", "violet"] + COLORS[4:]
    # LINESTYLE = LINESTYLE[:4] + [":", ":", ":"] + LINESTYLE[4:]
    # LINEWIDTH = LINEWIDTH[:4] + [DEFAULT_LINEWIDTH, DEFAULT_LINEWIDTH, DEFAULT_LINEWIDTH] + LINEWIDTH[4:]

    SYSTEMS = SYSTEMS[:4] + ["Lucie2.10_65k"] + SYSTEMS[4:]
    COLORS = COLORS[:4] + ["r"] + COLORS[4:]
    LINESTYLE = LINESTYLE[:4] + [":"] + LINESTYLE[4:]
    LINEWIDTH = LINEWIDTH[:4] + [DEFAULT_LINEWIDTH] + LINEWIDTH[4:]

RESULTS_DATAFRAME = None


def dataset_label(name):
    f = name.split(":")
    if len(f) > 2:
        f = f[:2]
    if "parquet" in f[-1]:
        f = f[:-1]
    if "Stack" in f[0]:
        f = f[1:]
    res = ".".join(f)
    res = res.rstrip("0123456789.")
    res = re.sub(r"(\w+)\.(\w+)", r"\1 (\2)", res)
    return res


def dataset_order(name, results_ref, datasets):
    lan_code = language_code(name)
    values = [
        pick_perf(results_ref, subname, "avg_length_token_char")
        for subname in datasets
        if dataset_label(subname).split(":")[0] == dataset_label(name).split(":")[0]
    ]
    values = [v for v in values if v is not None]
    score = np.mean(values) if values else 0
    if "Gallica" in name:
        score = 0
    if "Persee" in name:
        score = 0.01
    return (lan_code, score, datasets.index(name))


def language_code(name):
    if any(f in name for f in ["TheStack", "c++", "javascript", "python", "tex"]):  # Code
        return 10
    elif ":en" in name or ".en" in name:  # English
        return 1
    elif ":de" in name or ".de" in name:  # German
        return 3
    elif ":es" in name or ".es" in name:  # Spanish
        return 4
    elif ":it" in name or ".it" in name:  # Italian
        return 5
    return 2


def language_decode(code):
    return {
        2: "French",
        1: "English",
        3: "German",
        4: "Spanish",
        5: "Italian",
        10: "Code",
    }[code]


def collect_all_results(CSV_FILE="tokenizers_performances.csv"):
    perfs = {
        "Average n° characters per token (to be maximized)": "avg_length_token_char",
        # "Average Token Length without space": "avg_length_token_char_no_space",
        # "Average Token Length without space neither digit": "avg_length_token_char_no_space_no_digit",
        # "Tokens per Second": "tokens_per_second",
        "Fertility = Average n° tokens per word (to be minimized)": "avg_tokens_per_word",
    }

    with open(CSV_FILE) as f:
        results_df = pd.read_csv(f)

    if not PLOT_GALLICA_PERSEE:
        results_df = results_df[~results_df["dataset"].str.contains("Gallica|Persee")]

    if not PLOT_LUCIE_32k:
        results_df = results_df[~results_df["tokenizer"].str.contains("Lucie-32k")]

    results_df["tokens_per_second"] = results_df["# tokens"] / results_df["processing time"]
    results_df["avg_length_token_char"] = results_df["# characters"] / results_df["# tokens"]
    results_df["avg_tokens_per_word"] = results_df["# tokens"] / results_df["# words"]

    datasets = results_df["dataset"].unique().tolist()
    datasets = sorted(datasets, key=lambda x: dataset_order(x, results_df, datasets))
    systems = results_df["tokenizer"].unique().tolist()

    def key_tokenizer(system):
        perf = np.mean(
            [pick_perf(results_df[results_df["tokenizer"] == system], d, "avg_length_token_char") for d in datasets]
        )
        num_tokens = get_tokenizer_size(results_df, system)
        return (num_tokens, -perf)

    systems = sorted(systems, key=key_tokenizer, reverse=True)

    results = {}
    for system in systems:
        res = results_df[results_df["tokenizer"] == system]
        results[system] = res

    reference_system = "Lucie"
    limits = []
    limits_labels = []
    previous_lan = None
    for i, x in enumerate(datasets):
        lan = dataset_order(x, results[reference_system], datasets)[0]
        if lan != previous_lan:
            limits.append([limits[-1][1] if limits else 0, i + 1])
            limits_labels.append(language_decode(lan))
            previous_lan = lan
        else:
            limits[-1][1] = i + 1

    return results_df, results, datasets, perfs, limits, limits_labels


def pick_perf(res, dataset, perf_key):
    name_key = "name" if "name" in res.columns else "dataset"
    values = res[res[name_key] == dataset][perf_key].values
    if not len(values):
        return None
    return float(values[-1])


def get_tokenizer_size(results, name):
    size = results[results["tokenizer"] == name]["vocabulary size (tokens)"].values[-1]
    return size


def format_system(results, name):
    if SHOW_TOKENIZER_SIZE:
        size = get_tokenizer_size(results, name)
        size = str(round(size, 3) // 1000) + "k"
        return f"{name} ({size})"
    return name


if __name__ == "__main__":
    results_df, results, datasets, perfs, limits, limits_labels = collect_all_results()

    split_limits = True

    dataset_labels = [dataset_label(d) for d in datasets]

    nrows = len(perfs)
    ncols = 2 if SEPARATE_LAST_DATASETS else 1
    num_datasets_1 = 0
    num_datasets_2 = 0

    if SEPARATE_LAST_DATASETS:
        # 0.75 / 0.25 ratio for the columns
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(12, 8),
            gridspec_kw={
                "height_ratios": [0.5, 0.5],
                "width_ratios": [0.8, 0.2],
            },
            facecolor=(1, 1, 1, 0) if TRANSPARENT_BACKGROUND else (1, 1, 1),
        )
    else:
        plt.figure()

    # Set colorblind-friendly color palette
    color_palette = sns.color_palette("colorblind", n_colors=6)  # Colorblind-friendly palette

    for i_perf, (perf_label, perf_key) in enumerate(perfs.items()):
        ax = plt.subplot(nrows, ncols, i_perf * ncols + 1)
        for isystem, (system, res) in enumerate(results.items()):
            if res is None:
                continue
            # print(system)
            # print(res)
            vals = [pick_perf(res, d, perf_key) for d in datasets]
            # plt.bar(dataset_labels, vals, label = system)
            reference = "Lucie" not in system
            linewidth = LINEWIDTH[isystem] if isystem < len(LINEWIDTH) else (DEFAULT_LINEWIDTH)
            linestyle = LINESTYLE[isystem] if isystem < len(LINESTYLE) else ("-" if reference else "--")
            alpha = ALPHAS[isystem] if isystem < len(ALPHAS) else 1
            color = COLORS[isystem]
            if NUM_MORE_THAN_32K and isystem == 0:
                plt.plot([], [], color="white", label="> 32k tokens:")
            if isystem == NUM_MORE_THAN_32K:
                plt.plot([], [], color="white", label="32k tokens:")
            kwargs_plot = dict(
                label=format_system(results_df, system),
                marker="+",
                markersize=linewidth,
                linewidth=linewidth,
                linestyle=linestyle,
                color=color,
                alpha=alpha,
            )
            if split_limits:
                num_datasets_done = 0
                offset = 0
                for i_dataset_group, ((a, b), limit_label) in enumerate(zip(limits, limits_labels)):
                    num_datasets = len(datasets)
                    if SEPARATE_LAST_DATASETS:
                        if i_dataset_group == len(limits) - SEPARATE_LAST_DATASETS:
                            num_datasets_2 = len(datasets) - a
                            ax = plt.subplot(nrows, ncols, i_perf * ncols + 2)
                            offset = num_datasets_1 = num_datasets_done
                            num_datasets = num_datasets_2
                        else:
                            ax = plt.subplot(nrows, ncols, i_perf * ncols + 1)
                            offset = 0
                            num_datasets = num_datasets_done
                    # Re-equilibrate the legend
                    # if isystem == 4 and "label" in kwargs_plot:
                    #     plt.plot([], [], color="w", label=" ")
                    plt.plot(range(a - offset, b - offset), vals[a:b], **kwargs_plot)
                    kwargs_plot.pop("label", None)
                    plt.xlim(0, num_datasets - 1)
                    if isystem == 0:
                        from matplotlib.font_manager import FontProperties

                        font = FontProperties().copy()
                        # font.set_family('Ubuntu Mono')

                        num_datasets_final = limits[-1][1]
                        if SEPARATE_LAST_DATASETS:
                            if offset:
                                num_datasets_final = num_datasets_2
                            else:
                                num_datasets_final = limits[-2][1]
                        if i_dataset_group == len(limits) - 1:
                            num_datasets_final = limits[-1][1]

                        plt.text(
                            # (a+b-1)/2, min(vals),
                            (a + b - 1 - offset) / (2 * (num_datasets_final - 1)),
                            0.02,
                            limit_label,
                            fontdict=dict(fontsize=FONT_SIZE),
                            ha="center",
                            va="bottom",
                            transform=ax.transAxes,
                            fontproperties=font,
                        )
                    num_datasets_done += b - a
            else:
                plt.plot(vals, **kwargs_plot)

        for i_col in range(ncols):
            plt.subplot(nrows, ncols, i_perf * ncols + 1 + i_col)
            if i_col == 0:
                plt.title(perf_label, fontsize=FONT_SIZE)
            if i_perf == len(perfs) - 1:
                ticks = dataset_labels
                if i_col == 0:
                    plt.xlabel("Datasets", fontsize=FONT_SIZE)
            else:
                ticks = ["" for d in datasets]
            xticks = range(len(datasets))
            if num_datasets_2:
                if i_col == 0:
                    xticks = list(xticks)[:-num_datasets_2]
                    ticks = ticks[:-num_datasets_2]
                else:
                    xticks = list(xticks)[-num_datasets_2:]
                    xticks = [x - num_datasets_1 for x in xticks]
                    ticks = ticks[-num_datasets_2:]
            plt.xticks(xticks, ticks, rotation=100 * 2 / 3, ha="right", fontsize=int(FONT_SIZE * 12 / 16))
            if i_perf == len(perfs) - 1:
                if i_col == 0:
                    plt.legend(ncol=2, loc="upper left", fontsize=FONT_SIZE)
            plt.yticks(fontsize=FONT_SIZE)

    if RESULTS_DATAFRAME is not None:
        # Sort rows
        RESULTS_DATAFRAME = RESULTS_DATAFRAME.sort_values(
            by=["vocabulary size (tokens)", "tokenizer", "language", "dataset"]
        )
        RESULTS_DATAFRAME.to_csv("../chronicles/tokenization/tokenizers_performances.csv", index=False)

    plt.show()
