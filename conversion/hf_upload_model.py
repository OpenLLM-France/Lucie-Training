import json
import os
from pathlib import Path
from typing import Optional

import huggingface_hub

wd = Path(__file__).parent.parent.resolve()


def upload_to_huggingface_hub(
    repo_id: str,
    input_dir: Path,
    message: Optional[str] = None,
    revision: Optional[str] = None,
    format_json: Optional[bool] = False,
    create_repo: Optional[bool] = None,
):
    """Uploads a directory to Hugging Face Hub.

    Args:
        repo_id: The repository ID. For instance, if the URL of the repository is
            `https://huggingface.co/username/my-model`, the `repo_id` is `username/my-model`.
        input_dir: The directory to upload.
        message (Optional[str], optional): The commit message. Defaults to None.
        revision (Optional[str], optional): The revision to create. Defaults to None.
        format_json (Optional[bool], optional): Whether to ensure json files are not on one line. Defaults to False.
        create_repo (Optional[bool], optional): If None, will automatically create the repo if it doesn't exist.
    """

    repo_url = f"https://huggingface.co/{repo_id}"

    if format_json:
        format_json_files(input_dir)

    print(f"Uploading repository {repo_url} with:\n" + "\n".join(os.listdir(input_dir)))
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
        print(f"Creating repository {repo_url}")
        api.create_repo(
            repo_id=repo_id,
            private=True,
            repo_type="model",
            exist_ok=False,
        )

    if revision:
        print(f"Creating branch {revision} in {repo_url}")
        api.create_branch(repo_id, repo_type="model", branch=revision)
        # api.create_tag(repo_id, repo_type="model", revision=revision, tag=revision, tag_message=message)

    print("Pushing changes")
    api.upload_folder(
        folder_path=input_dir,
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=["lit_*", "pytorch_model.bin", "__pycache__"],
        revision=revision,
        commit_message=message,
    )


def is_hf_logged_in():
    try:
        huggingface_hub.HfApi().whoami()
        return True
    except Exception:
        return False


def format_json_files(file_or_folder, verbose=True):
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


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(upload_to_huggingface_hub)
