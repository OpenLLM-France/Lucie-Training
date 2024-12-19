import json
import math
import os
import re
import shutil
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm


def extract_events(logdir, subsample_factor=None, verbose=True):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    if verbose:
        print("Extracting events from log directory:", logdir)
    event_acc = EventAccumulator(logdir)
    if verbose:
        print("Loading events...")
    event_acc.Reload()
    if verbose:
        print("Events loaded.")

    tag_training_loss = "token/lm-loss-training/lm loss vs tokens"
    tag_tokens_sec = "iteration-time/tokens per second"
    tag_parallelization_factor = "step/world-size/world-size"
    tag_lr = "step/learning-rate/learning-rate"

    tags = sorted(event_acc.Tags()["scalars"])
    # print("Possible tags are:")
    # for tag in tags:
    #     print(f" - {tag}")
    for tag in [tag_training_loss, tag_tokens_sec, tag_parallelization_factor, tag_lr]:
        assert tag in tags, f"Tag '{tag}' not found in the event file (possible: {tags})"

    events = event_acc.Scalars(tag_training_loss)
    tokens = [e.step for e in events]
    losses = [e.value for e in events]

    time_events = [e.value for e in event_acc.Scalars(tag_tokens_sec)]
    parallel_events = [e.value for e in event_acc.Scalars(tag_parallelization_factor)]
    learning_rates = event_acc.Scalars(tag_lr)
    steps = [e.step for e in learning_rates]
    learning_rates = [e.value for e in learning_rates]
    assert len(time_events) == len(tokens)
    assert len(parallel_events) == len(tokens)
    assert len(learning_rates) == len(tokens)
    walltimes = []
    gputimes = []
    walltime_accu = 0
    gputime_accu = 0
    previous_tok = 0

    do_subsample = subsample_factor and subsample_factor > 1

    warning_given = False
    last_warning = None
    new_steps = []
    new_tokens = []
    new_losses = []
    new_learning_rates = []
    loss_accu = 0
    num_accu = 0
    steps_with_problems_tokens = []
    for i, (time, tok, parallel, step, loss, lr) in enumerate(
        zip(time_events, tokens, parallel_events, steps, losses, learning_rates)
    ):
        tok_per_sec = time
        if tok <= previous_tok:
            warning_msg = f"Skipping token {tok} <= {previous_tok}"
            if not warning_given:
                print("> " + warning_msg)
                warning_given = True
                steps_with_problems_tokens.append(tok)
            else:
                last_warning = warning_msg
            continue
        if warning_given:
            if last_warning:
                print("< " + last_warning)
            warning_given = False
            last_warning = None
        time = ((tok - previous_tok) / tok_per_sec) / 3600
        previous_tok = tok
        walltime_accu += time
        gputime_accu += time * parallel
        loss_accu += loss
        num_accu += 1
        if do_subsample and (i + 1) % subsample_factor != 0 and i != len(tokens) - 1:
            continue
        walltimes.append(walltime_accu)
        gputimes.append(gputime_accu)
        new_steps.append(step)
        new_tokens.append(tok)
        new_learning_rates.append(lr)
        new_losses.append(loss_accu / num_accu)
        loss_accu = 0
        num_accu = 0

    if warning_given:
        if last_warning:
            print("< " + last_warning)

    steps = new_steps
    tokens = new_tokens
    losses = new_losses
    learning_rates = new_learning_rates

    assert len(steps) == len(tokens), f"len(steps)={len(steps)} != len(tokens)={len(tokens)}"
    assert len(steps) == len(losses), f"len(steps)={len(steps)} != len(losses)={len(losses)}"
    assert len(steps) == len(walltimes), f"len(steps)={len(steps)} != len(walltimes)={len(walltimes)}"
    assert len(steps) == len(gputimes), f"len(steps)={len(steps)} != len(gputimes)={len(gputimes)}"
    assert len(steps) == len(learning_rates), f"len(steps)={len(steps)} != len(learning_rates)={len(learning_rates)}"

    return pd.DataFrame(
        {
            "training_steps": steps,
            "training_tokens": tokens,
            "training_loss": losses,
            "walltime": walltimes,
            "gputime": gputimes,
            "learning_rate": learning_rates,
        }
    ), {
        "steps_with_problems_tokens": steps_with_problems_tokens,
    }


def subsample_events(input_logdir, output_logdir, subsample_factor=10):
    assert subsample_factor and subsample_factor > 1
    if not os.path.exists(output_logdir):
        os.makedirs(output_logdir)

    for event_file in tqdm.tqdm(sorted(os.listdir(input_logdir))):
        if "events.out.tfevents" not in event_file:
            continue

        input_file_path = os.path.join(input_logdir, event_file)
        output_file_path = os.path.join(output_logdir, event_file)
        if os.path.exists(output_file_path):
            continue
        os.makedirs(output_logdir, exist_ok=True)

        try:
            subsample_events_file(input_file_path, output_file_path, subsample_factor)
        except (Exception, KeyboardInterrupt) as err:
            if os.path.exists(output_file_path):
                os.remove(output_file_path)
            raise RuntimeError(
                f"Error while subsampling events from '{input_file_path}' to '{output_file_path}'"
            ) from err


def subsample_events_file(input_file_path, output_file_path, subsample_factor):
    import tensorflow as tf

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    tmp_folder = tempfile.mkdtemp()

    event_acc = EventAccumulator(input_file_path)
    event_acc.Reload()

    min_wall_time = 1e32
    max_wall_time = -1

    try:
        with tf.summary.create_file_writer(tmp_folder).as_default() as writer:
            tags = event_acc.Tags()["scalars"]

            for tag in tags:
                use_max = False  # any(keyword in tag for keyword in ["time", "second"])
                accumulated_value = -1 if use_max else 0
                num_accu = 0

                events = event_acc.Scalars(tag)

                for i, event in enumerate(events):
                    wt = event.wall_time
                    min_wall_time = min(min_wall_time, wt)
                    max_wall_time = max(max_wall_time, wt)

                    num_accu += 1
                    if use_max:
                        accumulated_value = max(accumulated_value, event.value)
                    else:
                        accumulated_value += event.value

                    if (i + 1) % subsample_factor == 0 or i == len(events) - 1:
                        accumulated_value = accumulated_value if use_max else (accumulated_value / num_accu)
                        tf.summary.scalar(tag, accumulated_value, step=event.step)
                        writer.flush()

                        # Reset the accumulator
                        accumulated_value = -1 if use_max else 0
                        num_accu = 0

            if accumulated_value:
                accumulated_value = accumulated_value if use_max else (accumulated_value / num_accu)
                tf.summary.scalar(tag, accumulated_value, step=event.step)
                writer.flush()

    except (Exception, KeyboardInterrupt) as err:
        if os.path.exists(tmp_folder):
            shutil.rmtree(tmp_folder)
        raise RuntimeError(f"Error while subsampling events from '{input_file_path}' to '{output_file_path}'") from err

    files = [os.path.join(tmp_folder, fn) for fn in os.listdir(tmp_folder)]
    assert len(files) == 1, f"Expected only one file in the temporary folder, got {list(os.listdir(tmp_folder))}"
    shutil.move(files[0], output_file_path)
    os.rmdir(tmp_folder)

    time_total = (max_wall_time - min_wall_time) / 3600
    print(f"Collected stats from {time_total:.2f} hours total")


def format_big_integer(x):
    if x < 100:
        return str(int(x))
    if x < 500_000 and can_be_rounded(x, 1000):
        return f"{x / 1_000:.0f}K"
    if x < 500_000 and can_be_rounded(x, 100):
        return f"{x / 1_000:.1f}K"
    if x < 500_000:
        return f"{x / 1_000:.2f}K"
    if x < 500_000_000 and can_be_rounded(x, 1_000_000):
        return f"{x / 1_000_000:.0f}M"
    if x < 500_000_000 and can_be_rounded(x, 1_000_00):
        return f"{x / 1_000_000:.1f}M"
    if x < 500_000_000:
        return f"{x / 1_000_000:.2f}M"
    if x < 500_000_000_000 and can_be_rounded(x, 1_000_000_000):
        return f"{x / 1_000_000_000:.0f}B"
    if x < 500_000_000_000 and can_be_rounded(x, 1_000_000_00):
        return f"{x / 1_000_000_000:.1f}B"
    if x < 500_000_000_000:
        return f"{x / 1_000_000_000:.2f}B"
    if can_be_rounded(x, 1_000_000_000_000):
        return f"{x / 1_000_000_000_000:.0f}T"
    return f"{x / 1_000_000_000_000:.2f}T"


def can_be_rounded(x, ratio):
    return abs(x / ratio) % 1 <= 0.05


def format_learning_rate(x):
    if x == 0:
        return "0"
    assert x > 0
    if x < 1:
        if abs(x - 10 ** math.log10(x)) / abs(x) < 0.01:
            return f"{x:.0e}"
        return f"{x:.1e}"
    return f"{x:.2f}"


def format_learning_rates(learning_rates):
    learning_rates = list(learning_rates)
    if len(learning_rates) <= 3:
        lr0 = learning_rates[1]
        learning_rates = [learning_rates[0]] + [lr0 / 100, lr0 / 10] + learning_rates[1:]
    return learning_rates, [format_learning_rate(x) for x in learning_rates]


def set_xticks_custom(
    ax, tokens, values, rounded_to=None, fontsize=16, verbose=True, x_offset=None, last_offset=None, unit="", show=True
):
    xmin = min(tokens)
    xmax = max(tokens)
    # At first, should set the limits ???
    ax.set_xlim(xmin, xmax)

    if values is None:
        if verbose:
            print("Disabling xticks")
        ax.set_xticks([])
    else:
        assert len(values)

        max_num_steps = 15
        min_value, max_value = values[0], values[-1]
        if x_offset:
            x_offset_actual = min_value
            min_value -= x_offset_actual
            max_value -= x_offset_actual
        if rounded_to is None:
            minimum = (max_value - min_value) / 100
            rounded_to = 10 ** math.ceil(math.log10(minimum)) if minimum else 1
            multiply_by_five = True
            while (max_value - min_value) // rounded_to > max_num_steps:
                if multiply_by_five:
                    rounded_to *= 5
                else:
                    rounded_to *= 2
                multiply_by_five = not multiply_by_five

        assert values[0] >= 0, f"First value is {values[0]}"
        previous_rounded = values[0]  # // rounded_to * rounded_to

        if x_offset:
            xticks = [tokens[0]]
            if not xticks:
                xticks = [xmin]
            tick_str_offset = format_big_integer(x_offset) + unit
            xticks_string = ["0"]
            # xticks_string = [f"0\n({tick_str_offset} + …)" if last_offset is not None else "0"]
            previous_rounded -= x_offset_actual
        else:
            xticks = [0]
            xticks_string = ["0"]
            xmin = 0

        for tok, value in zip(tokens, values):
            if x_offset:
                value -= x_offset_actual
            new_rounded = value // rounded_to * rounded_to
            if new_rounded > previous_rounded:
                # Passed a new hours
                value = new_rounded
                if value == 0:
                    continue
                if value < 0:
                    import pdb

                    pdb.set_trace()
                    # There is a bug !
                tick_str = format_big_integer(value) + unit
                xticks.append(tok)
                xticks_string.append(tick_str)
                previous_rounded = new_rounded

        if True:  # last_offset:
            xticks.append(tokens[-1])
            tick_str_offset = format_big_integer(values[-1]) + unit
            xticks_string.append(f"\n{tick_str_offset}" + (" + …" if last_offset else ""))

        if verbose:
            print(
                f"Setting {len(xticks)} xticks rounded to {rounded_to}"
                f" in between {format_big_integer(min_value)} and {format_big_integer(max_value)} / {xmin=}, {xmax=}"
            )
        if not show:
            xticks_string = [" "] * len(xticks)
        ax.set_xticks(xticks, xticks_string, fontsize=fontsize, rotation=0, ha="left")

    # At last, set the limits
    ax.set_xlim(xmin, xmax)


def plot_convergence_curve(
    data,
    transparent=False,
    fontsize=16,
    switch_stage_tokens=None,
    problems=None,
    plot_learning_rate=True,
    all_stages_in_a_same_plot=False,
):
    all_tokens = np.array(data["training_tokens"])
    all_loss = np.array(data["training_loss"])
    all_learning_rates = np.array(data["learning_rate"])
    all_walltime = np.array(data["walltime"])
    all_gputime = np.array(data["gputime"])

    axes_names = ["# Tokens", "Training Time\n(hours)", "GPU hours"]
    nrows = len(axes_names) + (1 if plot_learning_rate else 0)
    ncols = 1 if all_stages_in_a_same_plot else (len(switch_stage_tokens) if switch_stage_tokens else 1)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6 * ncols, 7),
        gridspec_kw={
            "height_ratios": [10] + [0.2] * (nrows - 2) + [2],
            "width_ratios": [1 / ncols] * ncols,
        },
        facecolor=(1, 1, 1, 0) if transparent else (1, 1, 1),
    )
    # Adjust the space between the columns & rows
    plt.subplots_adjust(wspace=0.1, hspace=0.4)

    assert len(axes) == nrows
    if ncols == 1:
        # The output structure of plt.subplots depends on whether ncols>?1 ... :-(
        # So let's add a dummy dimension to the list,
        # to get an array (nrows x ncols) and treat everything the same below
        axes = [[ax] for ax in axes]

    fig_indices_main = [0]
    if not all_stages_in_a_same_plot and switch_stage_tokens:
        fig_indices_main = list(range(len(switch_stage_tokens)))

    default_min_num_tokens = -1
    default_max_num_tokens = all_tokens.max() + 1
    previous_points = {}
    for idx_fig in fig_indices_main:
        ax = axes[0][idx_fig]

        idx_first = 0
        idx_last = len(all_tokens)

        if switch_stage_tokens and not all_stages_in_a_same_plot:
            min_num_tokens = switch_stage_tokens.get(idx_fig, {}).get("first_token", default_min_num_tokens)
            max_num_tokens = switch_stage_tokens.get(idx_fig, {}).get("last_token", default_max_num_tokens)
            stage_name = switch_stage_tokens.get(idx_fig, {}).get("stage_name", "")
            stage_name = " ".join([w.capitalize() for w in stage_name.replace("_", " ").split()])
            assert (
                max_num_tokens > min_num_tokens
            ), f"Stage {stage_name}: max_num_tokens={max_num_tokens} <= min_num_tokens={min_num_tokens}"
            print(
                f"Stage {stage_name}: {format_big_integer(min_num_tokens)}"
                f" -> {format_big_integer(max_num_tokens)} tokens"
            )

            if min_num_tokens:
                # Find the first token
                for t in all_tokens:
                    if t > min_num_tokens:
                        break
                    idx_first += 1
            if max_num_tokens:
                # Find the last token
                for t in all_tokens[::-1]:
                    if t <= max_num_tokens:
                        break
                    idx_last -= 1

        tokens = all_tokens[idx_first:idx_last]
        loss = all_loss[idx_first:idx_last]
        learning_rates = all_learning_rates[idx_first:idx_last]
        walltime = all_walltime[idx_first:idx_last]
        gputime = all_gputime[idx_first:idx_last]

        xmin = min_num_tokens = tokens.min()
        xmax = max_num_tokens = tokens.max()

        # Title if subplots
        if ncols > 1 and stage_name:
            ax.set_title(re.sub(r"([0-9]) ", r"\1- ", stage_name), fontsize=fontsize)

        # - Plot the training loss
        ax.plot(
            tokens,
            loss,
            label="Training Loss",
            linewidth=4,
            color="blue",
        )
        if idx_fig == 0:
            ax.set_ylabel("-log Proba(next token)", fontsize=fontsize)
        # - Make the legend
        do_legend_yvalue = (
            "upper center"  # if (idx_fig == len(fig_indices_main) - 1) else "upper center" if idx_fig == 0 else None
        )
        # "upper right" if (idx_fig == len(fig_indices_main) - 1) else "lower left" if idx_fig == 0 else None
        # "best" if idx_fig in [0, len(fig_indices_main) - 1] else None
        do_legend_xticks = idx_fig in [0, len(fig_indices_main) - 1]
        if do_legend_yvalue:
            ax.legend(loc=do_legend_yvalue, fontsize=fontsize)
        # - Boundaries and Scales
        # ax.set_yscale('log')
        (ymin, ymax) = ax.get_ylim()
        # ymin = min(0, ymin)
        ymin = 1.2  # all_loss.min() * 0.95
        ymax = 2.4  # min(ymax, 2.4)
        ax.set_ylim(ymin, ymax)
        ax.grid(True, which="both", linestyle=":", linewidth=1)

        # (xmin, xmax) = ax.get_xlim()
        # xmin = max(min_num_tokens if idx_fig == 0 else tokens.min(), xmin)

        # Check the figure range represents everything
        assert xmin <= min_num_tokens, f"{xmin=}, {min_num_tokens=} for {stage_name=} {idx_fig=}"
        assert xmax >= max_num_tokens, f"{xmax=}, {max_num_tokens=} for {stage_name=} {idx_fig=}"

        last_offset = (
            idx_fig < len(switch_stage_tokens) - 1 and not all_stages_in_a_same_plot and len(switch_stage_tokens) > 1
        ) or None

        set_xticks_custom(
            ax,
            tokens,
            tokens,
            fontsize=fontsize,
            x_offset=previous_points.get("last_token"),
            last_offset=last_offset,
        )

        # Just for the font size... :-(
        losses_ticks = ax.get_yticks()
        ax.set_yticks(losses_ticks, [f"{x:.1f}" for x in losses_ticks], fontsize=fontsize)

        # Plot the tokens where problems occured
        if problems:
            for value in problems:
                if not (value >= min_num_tokens and value <= max_num_tokens):
                    continue
                ax.axvline(value, color="red", linestyle=":", linewidth=2)

        # Make the other x-scales (trainingtime and gputime)
        set_xticks_custom(
            axes[1][idx_fig],
            tokens,
            walltime,
            fontsize=fontsize,
            x_offset=previous_points.get("last_walltime"),
            last_offset=last_offset,
            # unit="H",
        )
        set_xticks_custom(
            axes[2][idx_fig],
            tokens,
            gputime,
            fontsize=fontsize,
            x_offset=previous_points.get("last_gputime"),
            last_offset=last_offset,
            # unit="H",
        )
        for iax, ax_name in enumerate(axes_names):
            sub_ax = axes[iax][idx_fig]
            if not do_legend_xticks:
                pass
            elif idx_fig > 0:
                sub_ax.set_xlabel(ax_name, fontsize=fontsize, ha="left")
                sub_ax.xaxis.set_label_coords(1.1, 0)
            else:
                sub_ax.set_xlabel(ax_name, fontsize=fontsize, ha="right")
                sub_ax.xaxis.set_label_coords(-0.1, 0)
            if iax > 0:
                # Remove upper, left and right axis
                for spine in "top", "right", "left":
                    sub_ax.spines[spine].set_visible(False)
                sub_ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)

        # Plot the learning rate
        if plot_learning_rate:
            ax_learning_rate = axes[nrows - 1][idx_fig]
            ax_learning_rate.plot(
                tokens,
                learning_rates,
                linewidth=4,
                color="orange",
                label="Learning Rate",
            )

            if do_legend_yvalue:
                ax_learning_rate.legend(loc=do_legend_yvalue.replace("upper", "lower"), fontsize=fontsize)
            ax_learning_rate.set_yscale("log")

            # Log-scale and ylim is ... buggy :-(
            # ymin = ymin + ymin/1000
            # ymax = ymax - ymin/1000
            # ymin = 1e-05 # 10 ** math.floor(math.log10(min_learning_rate))
            # ymax = 0.001 # 10 ** math.ceil(math.log10(max_learning_rate))
            ymin = 3e-06
            ymax = 3e-04

            ax_learning_rate.set_ylim(ymin, ymax)

            learning_rates_ticks = ax_learning_rate.get_yticks()

            set_xticks_custom(ax_learning_rate, tokens, tokens, show=False, x_offset=previous_points.get("last_token"))
            if idx_fig == 0:
                ax_learning_rate.set_yticks(*format_learning_rates(learning_rates_ticks), fontsize=fontsize)
            else:
                ax_learning_rate.set_yticks([])
            ax_learning_rate.set_ylim(ymin, ymax)
            ax_learning_rate.grid(True, which="both", linestyle=":", linewidth=1)
            # if idx_fig == 0:
            #     ax_learning_rate.set_ylabel("learning\nrate", fontsize=fontsize)

        # Update stats for next (sub)plot
        previous_points = {
            "last_token": max_num_tokens
            if max_num_tokens != default_max_num_tokens
            else previous_points.get("last_token", None),
            "last_walltime": walltime[-1],
            "last_gputime": gputime[-1],
        }

    # Deprecated (since subplot per stage is implemented):
    #   splot the stage switch by a vertical line between two consecutive stages (in a same plot)
    if switch_stage_tokens and all_stages_in_a_same_plot:
        assert ncols == 1
        for i_stage in range(len(switch_stage_tokens)):
            first_token = switch_stage_tokens[i_stage].get("first_token", 0)
            if first_token:
                for ax in [axes[0]] + ([axes[nrows - 1]] if plot_learning_rate else []):
                    ax.axvline(first_token, color="red", linestyle=":", linewidth=2)


if __name__ == "__main__":
    POSSIBLE_FOLDERS = [
        "/data-server/huggingface/models/OpenLLM-France/Lucie-7B/metadata/training_logs/tmp",
        "/mnt/d/home/jlouradour/src/OpenLLM/Bloom-NG-Training/assets/hugging_face/tmp",
    ]
    default_folder = POSSIBLE_FOLDERS[0]
    for f in POSSIBLE_FOLDERS:
        if os.path.exists(f):
            default_folder = f
            break

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("logdirs", type=str, help="Path to the TensorBoard log directory", nargs="+")
    parser.add_argument(
        "--subsample",
        type=int,
        help="Subsample factor for the events",
        default=25,
        # default=None,
    )
    parser.add_argument(
        "--accumulate",
        default=False,
        action="store_true",
        help="Accumulate the values on the same plot, if several logdir are given",
    )
    args = parser.parse_args()

    subsample_before = False  # Subsampling events does not work :-(

    has_create_csv = False
    complete_data = None
    complete_problems = {}
    switch_stage_tokens = {}
    for i_stage, logdir in enumerate(args.logdirs):
        stage_name = os.path.basename(logdir).replace("extension", "context extension")

        output_csv = logdir + (f"_subsampled{args.subsample}" if args.subsample else "") + ".csv"
        output_problems = logdir + (f"_subsampled{args.subsample}" if args.subsample else "") + "_problems.json"

        if not os.path.isfile(output_csv):
            if args.subsample and args.subsample > 1 and subsample_before:
                new_logdir = f"{logdir}_subsampled{args.subsample}"
                subsample_events(logdir, new_logdir, args.subsample)
                logdir = new_logdir

            data, problems = extract_events(logdir, args.subsample if not subsample_before else None)
            data.to_csv(output_csv, index=False)
            json.dump(problems, open(output_problems, "w"))  # , indent=2)
            has_create_csv = True

        data = pd.read_csv(output_csv)
        problems = json.load(open(output_problems))

        if complete_data is not None:
            # Merge this
            # {
            #     "training_steps": steps,
            #     "training_tokens": tokens,
            #     "training_loss": losses,
            #     "walltime": walltimes,
            #     "gputime": gputimes,
            #     "learning_rate": learning_rates,
            # }
            dict1 = complete_data.to_dict()
            dict2 = data.to_dict()
            last_event_idx = list(dict1["training_tokens"].keys())[-1]  # if len(dict1["training_tokens"]) else -1
            last_token = list(dict1["training_tokens"].values())[-1]
            last_step = list(dict1["training_steps"].values())[-1]
            switch_stage_tokens[i_stage] = {
                "stage_name": stage_name,
                "first_token": last_token,
            }
            switch_stage_tokens[i_stage - 1] = switch_stage_tokens[i_stage - 1] | {
                "last_token": last_token,
            }
            for key in dict1:
                accu = any(keyword in key for keyword in ["steps", "tokens", "walltime", "gputime"])
                last_value = dict1[key][last_event_idx]
                for event_idx, value in dict2[key].items():
                    if accu:
                        value += last_value
                    dict1[key][event_idx + last_event_idx + 1] = value
            complete_data = pd.DataFrame(dict1)
            for k in problems:
                assert k in complete_problems, f"Key {k} not found in complete_problems"
                complete_problems[k].extend([tok + last_token for tok in problems[k]])
        else:
            complete_data = data
            complete_problems = problems
            if len(args.logdirs) > 1:
                switch_stage_tokens[i_stage] = {"stage_name": stage_name, "first_token": 0}

    plot_convergence_curve(
        complete_data,
        switch_stage_tokens=switch_stage_tokens,
        # problems=complete_problems["steps_with_problems_tokens"],
        all_stages_in_a_same_plot=args.accumulate,
    )

    if not has_create_csv:
        plt.show()
