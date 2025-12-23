"""Download and prepare training datasets for translation model.

This script downloads parallel corpora from Hugging Face datasets.
You can modify this script to download other datasets.

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
    """Download English-Hindi parallel corpus from IITB."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install the datasets library:")
        print("  pip install datasets")
        return
    
    print("Downloading English-Hindi dataset...")
    ds = load_dataset("cfilt/iitb-english-hindi")
    
    # Convert to training format
    train_data = []
    
    for item in ds['train']:
        train_data.append({
            'source': item['translation']['en'],
            'target': item['translation']['hi']
        })
        
        # Limit to first 10000 samples for quick training
        if len(train_data) >= 10000:
            break
    
    # Save to processed data directory
    output_path = config.DATA_DIR / "processed" / "train_data.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(train_data)} samples to {output_path}")


def download_opus_dataset(src_lang: str = "en", tgt_lang: str = "es"):
    """Download parallel corpus from OPUS (optional).
    
    Args:
        src_lang: Source language code.
        tgt_lang: Target language code.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install the datasets library:")
        print("  pip install datasets")
        return
    
    print(f"Downloading {src_lang}-{tgt_lang} dataset from OPUS...")
    
    try:
        ds = load_dataset("opus100", f"{src_lang}-{tgt_lang}")
        
        train_data = []
        for item in ds['train']:
            train_data.append({
                'source': item['translation'][src_lang],
                'target': item['translation'][tgt_lang]
            })
            
            if len(train_data) >= 10000:
                break
        
        output_path = config.DATA_DIR / "processed" / f"train_data_{src_lang}_{tgt_lang}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(train_data)} samples to {output_path}")
        
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
    # download_opus_dataset("en", "es")  # English-Spanish
    # download_opus_dataset("en", "fr")  # English-French