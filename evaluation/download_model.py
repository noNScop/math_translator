#!/usr/bin/env python3
"""
Download a HuggingFace model to a local directory.

Usage:
    python evaluation/download_model.py
    python evaluation/download_model.py --model-id google/gemma-4-E4B-it --dest models/google/gemma-4-E4B-it
"""
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_ID = "google/gemma-4-E4B-it"
DEFAULT_DEST = PROJECT_ROOT / "models" / "google" / "gemma-4-E4B-it"


def download_model(model_id: str, dest: Path) -> None:
    if dest.exists() and any(dest.iterdir()):
        print(f"Model already exists at {dest}, skipping download.")
        return

    from huggingface_hub import snapshot_download
    print(f"Downloading {model_id} to {dest} ...")
    snapshot_download(repo_id=model_id, local_dir=str(dest))
    print(f"Done. Model saved to {dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a HuggingFace model locally")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args()

    download_model(args.model_id, args.dest)
