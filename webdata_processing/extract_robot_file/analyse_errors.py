import argparse
import os
import re

import matplotlib.pyplot as plt
import pandas as pd


def find_error(pattern, log):
    # Search for the pattern in the log string
    match = re.search(pattern, log)

    # If a match is found, return the error name
    if match:
        return match.group(1)
    else:
        return None


def find_connection_pool_error(log):
    pattern = r"ConnectionPool.*Caused by [\w\.].*(\[Errno ?.*\] .+)'\)\)"
    return find_error(pattern, log)


def find_timeout_error(log):
    pattern = r"ConnectionPool.*(Read timed out).*"
    return find_error(pattern, log)


def find_connect_timeout_error(log):
    pattern = r"ConnectionPool.*(connect timeout).*"
    return find_error(pattern, log)


def find_client_error(log):
    pattern = r"(4\d{2} Client Error).*"
    out = find_error(pattern, log)
    if out == "404 Client Error":
        return "404 Client Error"
    elif out:
        return "4xx Client Error"
    else:
        return None


def find_server_error(log):
    # Define the regex pattern
    pattern = r"(5\d{2} Server Error).*"
    if find_error(pattern, log):
        return "5xx Server Error"
    else:
        return None


def process_error(log):
    out = None
    if out is None:
        out = find_connection_pool_error(log)
    if out is None:
        out = find_timeout_error(log)
    if out is None:
        out = find_connect_timeout_error(log)
    if out is None:
        out = find_client_error(log)
    if out is None:
        out = find_server_error(log)
    if out is None:
        out = "Other error"
    return out


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str, default="out")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(args)
    os.makedirs(args.output_path, exist_ok=True)

    df = pd.read_csv(args.input_path)
    df["error_type"] = df["error"].apply(process_error)

    stats = df.groupby("error_type")["error_type"].size().reset_index(name="count")

    stats.plot.bar(x="error_type", y="count")
    plt.xticks(rotation=45, ha="right")
    plt.savefig(os.path.join(args.output_path, "error_type.png"), bbox_inches="tight")

    # What is other
    df[df["error_type"] == "Other error"].head(1000).to_csv(os.path.join(args.output_path, "other_error.csv"))
