import os
from pathlib import Path
from typing import Optional

import huggingface_hub

wd = Path(__file__).parent.parent.resolve()


def upload_to_huggingface_hub(
    repo_id: str,
    input_dir: Path,  # = wd / "hf_files" / "Claire-Falcon-7B-0.1",
    message: Optional[str] = None,
    revision: Optional[str] = None,
    create_repo: Optional[bool] = None,
):
    repo_url = f"https://huggingface.co/{repo_id}"

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


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(upload_to_huggingface_hub)
