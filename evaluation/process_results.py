import glob
import json
import re

import pandas as pd


def create_lucie_steps_to_tokens():
    lucie_steps_to_tokens = {
        753851: 3121737891840,
        5000: 5700059136,
        10000: 13884194816,
        15000: 26008354816,
        20000: 43747901440,
    }

    for i in range(1000):
        lucie_steps_to_tokens[20000 + i * 5000] = 43747901440 + i * 20971520000
    return lucie_steps_to_tokens


lucie_steps_to_tokens = create_lucie_steps_to_tokens()


def read_json(path):
    with open(path) as file:
        data = json.load(file)
    return data


def process_results(data):
    out = []
    for benchmark, results in data.items():
        metric = None
        if ("arc_" in benchmark) or ("hellaswag" in benchmark):
            metric = "acc_norm,none"
        elif (
            ("mmlu" == benchmark)
            or ("mmlu_continuation" == benchmark)
            or ("winogrande" == benchmark)
            or ("m_mmlu" in benchmark)
            or ("french_bench_grammar" == benchmark)
            or ("french_bench_vocab" == benchmark)
            or ("truthfulqa_mc2" == benchmark)
        ):
            metric = "acc,none"
        elif "gsm8k" == benchmark:
            metric = "exact_match,strict-match"
        elif (
            ("french_bench_fquadv2_genq" == benchmark)
            or ("french_bench_fquadv2_hasAns" == benchmark)
            or ("french_bench_multifquad" == benchmark)
            or ("french_bench_orangesum_abstract" in benchmark)
            or ("french_bench_trivia" == benchmark)
        ):
            metric = "rouge1,none"
        #
        if metric is not None:
            out.append({"benchmark": benchmark, "metric": metric, "score": results[metric]})

    return {"results": out}


def process_name(data):
    model_name = data["model_name"]
    out = {}
    if "Lucie" in model_name:  # Lucie model
        out["name"] = "lucie"
        out["checkpoint"] = False
        if "pretrained" in model_name:
            out["model_type"] = "pretraining"
            out["global_step"] = int(re.search(r"global_step(\d+)", model_name).group(1))
            out["num_tokens"] = lucie_steps_to_tokens[out["global_step"]]
            if out["global_step"] != 753851:
                out["checkpoint"] = True
        elif "extension" in model_name:
            out["model_type"] = "extension"
            out["num_tokens"] = lucie_steps_to_tokens[753851]
        elif "annealing" in model_name:
            out["model_type"] = "annealing"
            out["num_tokens"] = lucie_steps_to_tokens[753851]
        elif "instruction" in model_name:
            out["model_type"] = "instruction"
            out["num_tokens"] = lucie_steps_to_tokens[753851]
    else:
        name = model_name.split("/")[-1]
        out["name"] = name
        out["model_type"] = "pretraining"
        out["checkpoint"] = False
        if name == "Meta-Llama-3.1-8B":
            out["num_tokens"] = 15 * 10**12
        elif name == "Mistral-7B-v0.1":
            pass
        elif name == "bloom-7b1":
            out["num_tokens"] = 0.35 * 10**12
        elif name == "CroissantLLMBase":
            out["num_tokens"] = 3 * 10**12
        elif name == "falcon-7b":
            out["num_tokens"] = 3 * 10**12
        elif name == "pythia-6.9b":
            out["num_tokens"] = 299892736000
    return out


file_paths = glob.glob("out/**/*.json", recursive=True)

out = []

for file_path in file_paths:
    data = read_json(file_path)
    results = process_results(data.pop("results"))
    eval_metadata = {
        "model_name": data["model_name"],
        "chat_template": data["chat_template"] is not None,
        "fewshot_as_multiturn": data["fewshot_as_multiturn"],
    }
    out.append({**eval_metadata, **results})

df = pd.json_normalize(out)
df = df.groupby(["model_name", "chat_template", "fewshot_as_multiturn"])["results"].agg(sum).reset_index()
df = pd.concat([df, df.apply(lambda x: process_name(x), axis=1, result_type="expand")], axis=1)


def flatten_df(df):
    # Flatten the results column
    def flatten_results(row):
        # Convert the list of dictionaries in 'results' into a single dictionary
        flattened = {res["benchmark"]: res["score"] for res in row}
        return flattened

    # Apply flatten_results to the 'results' column and create new columns for each benchmark
    results_flattened = df["results"].apply(flatten_results)

    # Convert the resulting series of dictionaries into a DataFrame
    results_df = pd.DataFrame(results_flattened.tolist())

    # Concatenate the original DataFrame with the flattened results
    final_df = pd.concat([df.drop(columns=["results"]), results_df], axis=1)
    return final_df


df = flatten_df(df).fillna(-1)

df.to_csv("all_results.csv")
