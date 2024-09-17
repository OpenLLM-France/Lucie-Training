import yaml

import pandas as pd
import scipy

from typing import Dict, Tuple, Optional


def parse_dataset_stats(dataset_stats_path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Parse the dataset stats csv file in the assets folder
    """
    # Load the csv file
    stats = pd.read_csv(dataset_stats_path, sep=',')
    total_stats = stats.head(1)  # get the header for the total amount of tokens, etc
    total_stats.columns = total_stats.columns.str.lstrip()  # remove leading spaces in column names
    total_stats.columns = total_stats.columns.str.rstrip()  # remove trailing spaces in column names
    language_stats = stats.head(11).tail(10)  # first 11 rows are language stats + 1 header
    language_stats.columns = language_stats.columns.str.lstrip()
    language_stats.columns = language_stats.columns.str.rstrip()
    dataset_stats = stats.tail(len(stats) - 11)  # remove the first 12 rows
    dataset_stats.columns = dataset_stats.columns.str.lstrip()
    dataset_stats.columns = dataset_stats.columns.str.rstrip()

    return total_stats, language_stats, dataset_stats


def parse_weights_config(weights_config_path: str) -> Dict[Tuple[str, str], float]:
    """
    The weights config is stored in a yaml
    """
    data_dict = {}
    with open(weights_config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

        # Parse the content and populate the dictionary
        for key, value in config.items():
            # Split the key into language and domain
            language, domain = key.split('--')
            # Store the value in the dictionary with a tuple key
            data_dict[(language, domain)] = value

    return data_dict


def parse_eval_results(eval_results_path: str) -> Dict[str, Dict[str, float]]:
    """
    The eval results are stored in a csv file but there's some stuff to clean up
    to make it easy to exploit
    """
    eval_results = {}
    with open(eval_results_path, 'r') as f:
        lines = f.readline()
        # First line are the fields
        keys = lines.split(',')
        keys[-1] = keys[-1].replace('\n', '')
        # Rest of the lines are the results, load line by line and stop when done
        # First element is dataset and the rest are floats
        for line in f:
            values = line.replace("\"", "")
            values = values.replace("tensor(", "")
            values = values.replace(", device='cuda:0')", "")
            values = values.replace(", device='cuda:1')", "")
            values = values.split(',')
            values[0] = values[0].split('/')[-1]
            values[-1] = values[-1].replace('\n', '')
            # Store the results in a dictionary with the dataset as key and the values as another dictionary
            eval_results[values[0]] = {keys[i]: float(values[i]) for i in range(1, len(keys))}

    return eval_results


def compute_language_category_weights(category_df: pd.DataFrame, export_to_yml: str = None) -> pd.DataFrame:
    """
    Compute the sampling weights for each language-category pair to achieve the desired distribution
    in the blended dataset.

    Args:
        category_df (pd.DataFrame): DataFrame containing dataset information with columns 'category', 'name', 'language', and 'B tokens'.
        export_to_yml (str, optional): File path to export the weights to a YAML file. If None, no file is created.

    Returns:
        pd.DataFrame: DataFrame with columns 'language', 'category', and 'weight', where 'weight' indicates the proportion of
                      each dataset to sample to achieve the target distribution.
    """
    # Define the target distribution
    target_distribution = {
        'fr': 0.30,
        'en': 0.30,
        'code': 0.30,
        'others': 0.10  # For languages other than 'fr', 'en', and 'code'
    }

    # Clean up the language data to avoid case sensitivity issues
    category_df['language'] = category_df['language'].str.strip().str.lower()

    # Identify languages that fall under "others"
    known_languages = set(target_distribution.keys()) - {'others'}
    category_df['language_group'] = category_df['language'].apply(
        lambda x: x if x in known_languages else 'others'
    )

    # Aggregate the total number of tokens for each language-category pair in the category_df
    aggregated_tokens = category_df.groupby(['language', 'category'])['B tokens'].sum().reset_index()

    # Count the number of categories within each language
    category_counts = aggregated_tokens.groupby('language')['category'].count().reset_index()
    category_counts.columns = ['language', 'category_count']

    # Calculate the number of unique languages in the "others" group
    others_languages = category_df[category_df['language_group'] == 'others']['language'].unique()
    num_others_languages = len(others_languages)

    # Create a list to store the rows for the weights DataFrame
    weight_rows = []

    # Calculate the weights for each language-category pair
    for _, row in category_counts.iterrows():
        language = row['language']
        num_categories = row['category_count']

        if language in known_languages:
            # Distribute the known language's target proportion equally across its categories
            target_proportion = target_distribution[language] / num_categories
        else:
            # Distribute the "others" proportion equally across all "other" languages and their categories
            target_proportion = (target_distribution['others'] / num_others_languages) / num_categories

        # Extract the relevant rows from the aggregated_tokens for this language
        language_pairs = aggregated_tokens[aggregated_tokens['language'] == language]

        # Assign equal weight to each category within the language
        for _, pair in language_pairs.iterrows():
            weight_rows.append({
                'language': pair['language'],
                'category': pair['category'],
                'weight': target_proportion
            })

    # Convert the list of rows into a DataFrame
    weights_df = pd.DataFrame(weight_rows)

    # Normalize the weights so they sum to 1
    weights_df['weight'] = weights_df['weight'] / weights_df['weight'].sum() * 100. # Convert to percentage

    # Export to YAML if requested
    if export_to_yml:
        # Create a dictionary in the format "language--category": weight
        weight_dict = {f"{row['language']}--{row['category']}": row['weight'] for _, row in weights_df.iterrows()}

        # Write the dictionary to a YAML file
        with open(export_to_yml, 'w') as yml_file:
            yaml.dump(weight_dict, yml_file, default_flow_style=False)

    return weights_df


def rank_datasets_by_PPL(eval_results: Dict[str, Dict[str, float]]):
    """
    Rank the datasets by PPL
    """
    # Sort the datasets by PPL
    sorted_datasets = sorted(eval_results.items(), key=lambda x: x[1]['PPL'])
    # Create a dictionary with the dataset as key and the rank as value
    rank = {dataset[0]: i for i, dataset in enumerate(sorted_datasets, 1)}

    return rank


def compute_spearman_correlation(ranking_0, ranking_1):
    """
    Compute the spearman correlation of the rankings of datasets by PPL for two different models
    """
    # Compute the spearman correlation
    assert set(ranking_0.keys()) == set(ranking_1.keys()), "Rankings must have the same domains."

    # Extract the rankings into lists
    ranking_list_1 = [ranking_0[domain] for domain in sorted(ranking_0.keys())]
    ranking_list_2 = [ranking_1[domain] for domain in sorted(ranking_1.keys())]

    # Compute Spearman correlation
    correlation, p_value = scipy.stats.spearmanr(ranking_list_1, ranking_list_2)

    return correlation, p_value


def compute_tokens_seen_from_dataset(dataset_stats, domain_proportions, n_steps: int, batch_size: int):
    """
    Compute the number of tokens seen from a dataset after n_steps at a given batch size
    """
    # Compute the total number of tokens in each dataset
    print(dataset_stats.keys())
    tokens_per_dataset = dataset_stats.set_index('name')['B tokens'].to_dict()
    tokens_per_dataset = {k.rstrip(): v for k, v in tokens_per_dataset.items()}
    print(tokens_per_dataset)
    exit()

    # TODO compute % of tokens from dataset for each domain
    # TODO estimate the number of tokens after n_steps
    # TODO maybe barplot with results ?


    # Compute the number of tokens seen from the dataset after n_steps
    tokens_seen = 0
    for domain, proportion in domain_proportions.items():
        tokens_seen += proportion * total_tokens * n_steps * batch_size

    return tokens_seen



if __name__ == '__main__':
    total_stats, language_stats, dataset_stats = parse_dataset_stats('../assets/stats_datasets_web.csv')
    domain_proportions = parse_weights_config("../training/domain_proportions.yml")
    eval_results = parse_eval_results("./perplexity.csv")

    # Compute the domain proportions for each language
    domain_weights = compute_language_category_weights(dataset_stats,
                                                       export_to_yml='../ablation/data_config/config01.yml')
    print(domain_weights)

    correlation, p_value = compute_spearman_correlation(
        rank_datasets_by_PPL(eval_results),
        rank_datasets_by_PPL(eval_results)
    )
    print(f"Spearman correlation between rankings: {correlation} with p-value: {p_value}")

    # compute_tokens_seen_from_dataset(dataset_stats, domain_proportions, 1_00_000, 256)
