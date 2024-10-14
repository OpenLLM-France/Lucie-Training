import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Literal, Optional

import huggingface_hub

wd = Path(__file__).parent.resolve()

_readme_file_main = wd / "README.md"
_readme_file_optimizer = wd / "README_optimizer.md"
_readme_header_file = wd / "README_header.yaml"
_model_config_files = [wd / "config.json", wd / "generation_config.json"]
for fn in [_readme_file_main, _readme_file_optimizer, _readme_header_file] + _model_config_files:
    assert fn.exists(), f"File not found at {fn}"


def upload_to_huggingface_hub(
    repo_id: str,
    input: Path,
    message: Optional[str] = None,
    training_steps: Optional[str] = None,
    type: Literal["final", "checkpoint", "optimizer", "tokenizer", "init"] = "final",
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
        message (Optional[str], optional): The commit message. Defaults to None.
        training_steps (Optional[int], optional): The number of the step. Defaults to None.
        is_checkpoint (bool): Whether the upload is a checkpoint. Defaults to False.
        format_json (bool): Whether to ensure json files are not on one line. Defaults to False.
        create_repo (Optional[bool], optional): If None, will automatically create the repo if it doesn't exist.
    """

    assert os.path.exists(input), f"Input {input} must be an existing file or directory"
    try:
        training_steps = int(training_steps)
    except (TypeError, ValueError):
        pass
    is_checkpoint = type in ["checkpoint", "optimizer"]
    is_optimizer = type == "optimizer"
    is_tokenizer = type == "tokenizer"
    dump_readme = (not is_checkpoint or is_optimizer) and not is_tokenizer
    if is_checkpoint and not is_optimizer:
        assert (
            isinstance(training_steps, int) and training_steps >= 0
        ), "Training steps must be provided for a checkpoint"

    upload_folder = input if os.path.isdir(input) else None
    upload_files = [input] if not upload_folder else []
    repo_url = f"https://huggingface.co/{repo_id}"

    tmp_files = []
    config_and_readme_folder = input if add_files_in_folder else tempfile.gettempdir()

    if upload_folder:
        # Create the README.md file
        readme_content = "---\n"
        with open(_readme_header_file) as f:
            readme_content += f.read().strip() + "\n"
        if isinstance(training_steps, int) and training_steps >= 0:
            readme_content += model_yaml_footer(training_steps).strip() + "\n"
        readme_content += "---\n"
        if dump_readme:
            readme_model_file = _readme_file_optimizer if is_optimizer else _readme_file_main
            with open(readme_model_file) as f:
                readme_content += "\n" + f.read().strip() + "\n"
        tmp_file = os.path.join(config_and_readme_folder, "README.md")
        if not add_files_in_folder:
            upload_files.append(tmp_file)
        with open(tmp_file, "w") as f:
            f.write(readme_content)

        # Copy config files
        # (add training progress metadata if training_steps is provided)
        if not is_tokenizer:
            for config_file in _model_config_files:
                config_filename = os.path.basename(config_file)
                if is_optimizer and config_filename in ["generation_config.json"]:
                    continue
                tmp_file = os.path.join(config_and_readme_folder, config_filename)
                if not add_files_in_folder:
                    upload_files.append(tmp_file)
                if isinstance(training_steps, int) and training_steps >= 0 and config_filename == "config.json":
                    with open(config_file) as f:
                        config = json.load(f)
                    config["training_steps"] = training_steps
                    config["training_tokens"] = training_step_to_tokens(training_steps)
                    with open(tmp_file, "w") as f:
                        json.dump(config, f, indent=2)
                else:
                    shutil.copy2(config_file, tmp_file)

        # Will remove files that were temporary created
        tmp_files = upload_files

    try:
        if format_json:
            format_json_files(input)

        if not is_hf_logged_in():
            huggingface_hub.login()

        api = huggingface_hub.HfApi()

        if create_repo is None:
            create_repo = False
            try:
                api.repo_info(repo_id)
            except huggingface_hub.utils.RepositoryNotFoundError:
                create_repo = True

        if create_repo:
            if not message:
                message = "initial commit"
            print(f"Create repository {repo_url}")
            api.create_repo(
                repo_id=repo_id,
                private=True,
                repo_type="model",
                exist_ok=False,
            )

        revision = None
        if isinstance(training_steps, int):
            revision = f"step{training_steps:07d}" if (is_checkpoint and training_steps >= 0) else None
        elif training_steps:
            revision = training_steps

        is_branch_new = False
        revision_info = ""
        if revision:
            revision_info = f" (branch {revision})"
            try:
                api.create_branch(repo_id, repo_type="model", branch=revision)
                is_branch_new = True
            except huggingface_hub.utils._errors.HfHubHTTPError:
                pass
            if is_branch_new:
                print(f"Create branch {revision} in {repo_url}")
            # api.create_tag(repo_id, repo_type="model", revision=revision, tag=revision, tag_message=message)

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
                    ["__pycache__"]  # , "pytorch_model.bin" ?
                ]
                if is_optimizer:
                    # Send in several parts (because there are many big files)
                    all_patterns = ["layer*", "bf16*", "mp_rank*", "*weight", "optimizer_state.pt"]
                    ignore_patterns_list = [
                        ignore_patterns_list[0] + list(set(all_patterns) - {keep_pattern})
                        for keep_pattern in all_patterns
                    ]
                for ignore_patterns in ignore_patterns_list:
                    content_filtered = []
                    for root, _, files in os.walk(input):
                        for file in files:
                            if not any(re.match(p.replace("*", ".*") + r"$", file) for p in ignore_patterns):
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
                    api.upload_folder(**kwargs, ignore_patterns=ignore_patterns)
                    uploaded_something = True

            upload_folder = uploaded_something

        for upload_file in upload_files:
            filename = os.path.basename(upload_file)
            print(f"Update repository {repo_url}{revision_info} with file {filename}")
            if not message or upload_folder:
                message = "{} {}".format(
                    "Upload" if (create_repo and not is_branch_new) else "Update", os.path.splitext(filename)[0]
                )
            api.upload_file(
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


def format_json_files(file_or_folder, verbose=True):
    """
    Format json files that are on one line.
    """
    if file_or_folder.endswith(".json") and os.path.is_file(file_or_folder):
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


def training_step_to_tokens(training_steps):
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
    return training_tokens


def model_yaml_footer(training_steps, context_length=4096):
    return f"""
training_progress:
   num_steps: {training_steps}
   num_tokens: {training_step_to_tokens(training_steps)}
   context_length: {context_length}
"""


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(upload_to_huggingface_hub)
