import json
import os
import re
from pathlib import Path
from typing import Optional

import huggingface_hub

wd = Path(__file__).parent.resolve()


def upload_to_huggingface_hub(
    repo_id: str,
    input: Path,
    message: Optional[str] = None,
    training_steps: Optional[str] = None,
    training_phase: Optional[str] = None,
    is_optimizer: bool = False,
    format_json: bool = False,
    create_repo: Optional[bool] = None,
    add_files_in_folder: bool = False,
):
    """Uploads a directory to Hugging Face Hub.

    Args:
        repo_id: The repository ID. For instance, if the URL of the repository is
            `https://huggingface.co/username/my-model`, the `repo_id` is `username/my-model`.
        input: Directory or file to upload.
        message (str, optional): The commit message. Defaults to None.
        training_steps (int, optional): The number of the step. Defaults to None.
        training_phase (str, optional): The prefix of the revision (ex: extension_). Defaults to None.
        is_checkpoint (bool): Whether the upload is a checkpoint. Defaults to False.
        format_json (bool): Whether to ensure json files are not on one line. Defaults to False.
        create_repo (Optional[bool], optional): If None, will automatically create the repo if it doesn't exist.
    """

    assert os.path.exists(input), f"Input {input} must be an existing file or directory"
    try:
        training_steps = int(training_steps)
    except (TypeError, ValueError):
        pass
    type = "tokenizer"
    is_checkpoint = type in ["checkpoint", "optimizer"]
    is_optimizer = type in ["optimizer", "final_optimizer"]
    # is_tokenizer = type == "tokenizer"
    # dump_readme = (not is_checkpoint or is_optimizer) and not is_tokenizer
    if is_checkpoint and not is_optimizer:
        assert (
            isinstance(training_steps, int) and training_steps >= 0
        ), "Training steps must be provided for a checkpoint"

    upload_folder = input if os.path.isdir(input) else None
    upload_files = [input] if not upload_folder else []
    repo_url = f"https://huggingface.co/{repo_id}"

    tmp_files = []
    # config_and_readme_folder = input if add_files_in_folder else tempfile.gettempdir()

    if upload_folder:
        # Will remove files that were temporary created
        tmp_files = upload_files

    try:
        if format_json:
            format_json_files(input)

        hf_api, repo_created = connect_to_huggingface(repo_id, create_repo)
        if not message and repo_created:
            message = "initial commit"

        revision = None
        if isinstance(training_steps, int):
            revision = f"step{training_steps:07d}" if (is_checkpoint and training_steps >= 0) else None
        elif training_steps:
            revision = training_steps

        if training_phase:
            assert revision, "A revision must be provided to use a revision prefix"
            revision = f"{training_phase}_{revision}"

        is_branch_new = False
        revision_info = ""
        if revision:
            revision_info = f" (branch {revision})"
            try:
                hf_api.create_branch(repo_id, repo_type="model", branch=revision)
                is_branch_new = True
            except Exception as err:  # huggingface_hub.utils._errors.HfHubHTTPError ?
                print(str(err).split("\n")[-1])
            if is_branch_new:
                print(f"Create branch {revision} in {repo_url}")
            # hf_api.create_tag(repo_id, repo_type="model", revision=revision, tag=revision, tag_message=message)

        if upload_folder:
            content = sorted(os.listdir(input))
            uploaded_something = False
            if content:
                kwargs = dict(
                    folder_path=input,
                    repo_id=repo_id,
                    repo_type="model",
                    revision=revision,
                    commit_message=message,
                )
                ignore_patterns_list = [
                    ["__pycache__", "eval.csv", "training_info.json", "*py"]  # , "pytorch_model.bin" ?
                ]
                if is_optimizer:
                    # Send in several parts (because there are many big files)
                    all_patterns = [
                        "layer*",
                        "bf16*",
                        "mp_rank*",
                        "**/**/fp32*",
                        "**/**/exp_avg*",
                        "**/optimizer_state.pt",
                    ]
                    ignore_patterns_list = [
                        ignore_patterns_list[0] + list(set(all_patterns) - {keep_pattern})
                        for keep_pattern in all_patterns
                    ]
                for ignore_patterns in ignore_patterns_list:
                    content_filtered = []
                    for root, _, files in os.walk(input):
                        for file in files:
                            if not any(
                                re.match(p.split("/")[-1].replace("*", ".*") + r"$", file) for p in ignore_patterns
                            ):
                                content_filtered.append(os.path.relpath(os.path.join(root, file), input))
                    content_filtered = sorted(content_filtered)
                    if not content_filtered:
                        print(f"Nothing to upload in {input} with {ignore_patterns=}")
                        continue
                    if len(content_filtered) > 20:
                        content_filtered = content_filtered[:10] + ["..."] + content_filtered[-10:]
                    print(
                        f"Update repository {repo_url}{revision_info} with containt of {os.path.realpath(input)}:"
                        + ("\n├── " if len(content_filtered) > 1 else "")
                        + "\n├── ".join(content_filtered[:-1])
                        + "\n└── "
                        + content_filtered[-1]
                    )
                    hf_api.upload_folder(**kwargs, ignore_patterns=ignore_patterns)
                    uploaded_something = True

            upload_folder = uploaded_something

        for upload_file in upload_files:
            filename = os.path.basename(upload_file)
            print(f"Update repository {repo_url}{revision_info} with file {filename}")
            if not message or upload_folder:
                message = "{} {}".format(
                    "Upload" if (repo_created and not is_branch_new) else "Update", os.path.splitext(filename)[0]
                )
            hf_api.upload_file(
                path_or_fileobj=upload_file,
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="model",
                revision=revision,
                commit_message=message,
            )

    finally:
        for tmp_file in tmp_files:
            os.remove(tmp_file)


def is_hf_logged_in():
    try:
        huggingface_hub.HfApi().whoami()
        return True
    except Exception:
        return False


def connect_to_huggingface(repo_id, create_repo=None, repo_type="model"):
    if not is_hf_logged_in():
        huggingface_hub.login()

    hf_api = huggingface_hub.HfApi()

    if create_repo is None:
        create_repo = False
        try:
            hf_api.repo_info(repo_id, repo_type=repo_type)
        except huggingface_hub.utils.RepositoryNotFoundError:
            create_repo = True

    if create_repo:
        repo_url = f"https://huggingface.co/{repo_id}"
        print(f"Create repository {repo_url}")
        hf_api.create_repo(
            repo_id=repo_id,
            private=True,
            repo_type=repo_type,
            exist_ok=False,
        )

    return hf_api, create_repo


def format_json_files(file_or_folder, verbose=True):
    """
    Format json files that are on one line.
    """
    if isinstance(file_or_folder, str):
        is_json = file_or_folder.endswith(".json") and os.path.isfile(file_or_folder)
    else:
        is_json = file_or_folder.suffix == ".json" and file_or_folder.is_file()
    if is_json:
        json_file = file_or_folder
        num_lines = sum(1 for line in open(json_file))
        if num_lines == 1:
            # Json file is on one line, so we format it (with "indent=2" to enforce at most 1 attribute per line)
            if verbose:
                print(f"Formatting {json_file}...")
            with open(json_file) as f:
                data = json.load(f)
            json_file_tmp = json_file + "_fmt.tmp"
            with open(json_file_tmp, "w") as f:
                json.dump(data, json_file_tmp, indent=2)
            os.rename(json_file_tmp, json_file)
    elif os.path.isdir(file_or_folder):
        for root, _, files in os.walk(file_or_folder):
            for file in files:
                if file.endswith(".json"):
                    format_json_files(os.path.join(root, file))
    else:
        if verbose:
            print(f"Ignoring {file_or_folder}...")


def training_step_total(training_steps, training_phase):
    if not training_phase:
        return training_steps

    if training_phase == "extension":
        return 753851 + training_steps

    raise ValueError(f"Unknown training phase {training_phase}")


def training_step_to_tokens(training_steps, training_phase=None):
    if not training_phase:
        training_tokens = {  # "pretrained": {
            0: 0,
            5000: 5700059136,
            10000: 13884194816,
            15000: 26008354816,
            # 20000: 43747901440,
            # 25000: 43747901440 + 20971520000, # 64719421440,
            # 30000: 43747901440 + 2 * 20971520000, # 85690941440,
            # 35000: 43747901440 + 3 * 20971520000, # 106662461440,
            # ...
            22818: 55567450112,
        }.get(training_steps)

        if training_tokens is None:
            assert training_steps >= 20000, f"Cannot infer number of tokens for {training_steps=}"
            training_tokens = round(43747901440 + ((training_steps - 20000) / 5000) * 20971520000)

    elif training_phase == "extension":
        num_steps_phase_1 = training_step_total(0, training_phase)
        num_tokens_phase_1 = training_step_to_tokens(num_steps_phase_1)
        training_tokens = {
            250: 1024000000,
            500: 2048000000,
            750: 3072000000,
            1000: 4096000000,
            1220: 4997120000,
        }.get(training_steps)
        assert training_tokens, f"Unknown training step {training_steps} for training phase {training_phase}"
        training_tokens += num_tokens_phase_1

    else:
        raise ValueError(f"Unknown training phase {training_phase}")

    return training_tokens


def model_yaml_footer(training_steps, training_phase=None):
    if not training_phase:
        context_length = 4096
    else:
        context_length = 32_000
    return f"""
training_progress:
   num_steps: {training_step_total(training_steps, training_phase)}
   num_tokens: {training_step_to_tokens(training_steps, training_phase)}
   context_length: {context_length}
"""


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(upload_to_huggingface_hub)
