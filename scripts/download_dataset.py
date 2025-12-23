"""
Download and prepare training datasets for translation model.

This script downloads parallel corpora from Hugging Face datasets
and saves them in a clean JSON format for training.

Usage:
    python scripts/download_dataset.py
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config


def download_en_hi_dataset():
    """Download FULL English-Hindi parallel corpus from IITB."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install the datasets library:")
        print("  pip install datasets")
        return

    print("Downloading FULL English-Hindi dataset (IITB)...")

    ds = load_dataset("cfilt/iitb-english-hindi")

    train_data = []

    for item in ds["train"]:
        train_data.append({
            "source": item["translation"]["en"],
            "target": item["translation"]["hi"]
        })

    output_path = config.DATA_DIR / "processed" / "train_data.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    print(f"Saved ALL {len(train_data)} samples to {output_path}")


def download_opus_dataset(src_lang: str = "en", tgt_lang: str = "es"):
    """Download FULL parallel corpus from OPUS-100.

    Args:
        src_lang: Source language code
        tgt_lang: Target language code
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install the datasets library:")
        print("  pip install datasets")
        return

    print(f"Downloading FULL {src_lang}-{tgt_lang} dataset from OPUS-100...")

    try:
        ds = load_dataset("opus100", f"{src_lang}-{tgt_lang}")

        train_data = []

        for item in ds["train"]:
            train_data.append({
                "source": item["translation"][src_lang],
                "target": item["translation"][tgt_lang]
            })

        output_path = (
            config.DATA_DIR / "processed" /
            f"train_data_{src_lang}_{tgt_lang}.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)

        print(f"Saved ALL {len(train_data)} samples to {output_path}")

    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Available language pairs: en-es, en-fr, en-de, en-zh, etc.")


if __name__ == "__main__":
    print("=" * 50)
    print("Dataset Downloader for Translation Model")
    print("=" * 50)

    # Download English-Hindi by default
    download_en_hi_dataset()

    # Uncomment to download other language pairs:
    # download_opus_dataset("en", "es")
    # download_opus_dataset("en", "fr")
