import argparse
import json
import os
import urllib.robotparser
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from tqdm import tqdm


def robot_extract(domain, prefix="https://", timeout=2):
    rp = urllib.robotparser.RobotFileParser()
    url = prefix + domain
    robots_url = url + "/robots.txt"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 \
        (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    }
    response = requests.get(robots_url, headers=headers, timeout=timeout, allow_redirects=True)
    response.raise_for_status()  # Raise an exception for HTTP errors
    rp.parse(response.text.splitlines())
    can_fetch = rp.can_fetch("CCBot", url)
    return can_fetch, response.text


def write_line(path, n_chunk, line):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, f"{n_chunk:05d}.jsonl"), "a", encoding="utf-8") as file:
        file.write(json.dumps(line, ensure_ascii=False) + "\n")


def process_domain(domain, output_dir, n_chunk):
    try:
        try:
            can_fetch, text = robot_extract(domain, prefix="https://")
        except Exception:
            can_fetch, text = robot_extract(domain, prefix="http://")
        # Save can_fetch in json
        write_line(f"{output_dir}/can_fetch", n_chunk, {"domain": domain, "can_fetch": can_fetch})
        # Save robots.txt
        write_line(f"{output_dir}/robots_txt_files", n_chunk, {"domain": domain, "text": text})
    except Exception as e:
        # Save error
        write_line(f"{output_dir}/logs", n_chunk, {"domain": domain, "error": str(e)})


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, help="Directory where the output will be stored.")
    parser.add_argument("--input_file", type=str, help="Input csv file containing the domains.")
    parser.add_argument("--input_field", type=str, default="url", help="Name of the field containing the domains.")
    parser.add_argument("--num_workers", type=int, default=20, help="Number of workers to use.")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(args)

    output_dir = args.output_path
    input_file = args.input_file
    input_field = args.input_field
    num_workers = args.num_workers
    chunk_size = 10000  # If you you change this during process it can break or create some incoherence

    # Path for saving and loading completed tasks
    os.makedirs(f"{output_dir}/completed_chunk", exist_ok=True)

    # Read the CSV file once
    list_of_domains = pd.read_csv(input_file)[input_field].values
    total_chunks = len(list_of_domains) // chunk_size + (1 if len(list_of_domains) % chunk_size != 0 else 0)

    for n_chunk in range(total_chunks):
        print(f"Chunk #{n_chunk}")
        if not os.path.exists(f"{output_dir}/completed_chunk/{n_chunk:05d}"):
            # Create the chunk
            domains = list_of_domains[n_chunk * chunk_size : (n_chunk + 1) * chunk_size]

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(process_domain, domain, output_dir, n_chunk) for domain in domains]
                for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching robots.txt"):
                    try:
                        future.result(timeout=60)  # Ensure exceptions are raised
                    except Exception as e:
                        print(f"Task failed: {e}")

        # Save in completed file
        with open(f"{output_dir}/completed_chunk/{n_chunk:05d}", "w") as file:
            pass
