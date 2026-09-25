from typing import TYPE_CHECKING


from huggingface_hub import snapshot_download
from src.llm.helpers.config import MODELS

if TYPE_CHECKING:
    from pathlib import Path


def download_model(model_name: str) -> str:
    return snapshot_download(repo_id=model_name, local_dir=MODELS / model_name)


def get_model_path(model_name: str) -> Path:
    return MODELS / (model_name)


def model_exists(model_name: str) -> bool:
    return get_model_path(model_name).exists()


def ensure_model(model_name: str):
    if not model_exists(model_name=model_name):
        return download_model(model_name=model_name)

    return get_model_path(model_name=model_name)
