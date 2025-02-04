import os
import shutil
from huggingface_hub import HfApi, Repository
import subprocess

datasets = [
    # "capitals",
    # "hemisphere",
    # "population",
    # "sciq",
    # "sentiment",
    # "nli",
    # "authors",
    # "addition",
    # "subtraction",
    # "multiplication",
    # "modularaddition",
    "squaring",
]

files_to_delete = [
    "config.json",
    "generation_config.json",
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
    "model.safetensors.index.json",
]

api = HfApi()

for dataset in datasets:
    repo_name = f"EleutherAI/Meta-Llama-3.1-8B-{dataset}-random-standardized-many-random-names"
    local_dir = f"Meta-Llama-3.1-8B-{dataset}"

    # Clone the repository
    Repository(local_dir, clone_from=repo_name)

    # Delete specified files
    deleted = False
    for file in files_to_delete:
        file_path = os.path.join(local_dir, file)
        if os.path.exists(file_path):
            deleted = True
            os.remove(file_path)

    if deleted:
        # Commit and push changes
        repo = Repository(local_dir)
        repo.git_add()
        repo.git_commit("Remove non peft components")
        repo.git_push()

    # Clean up local directory
    shutil.rmtree(local_dir)

print("All repositories processed successfully.")