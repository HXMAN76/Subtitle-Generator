"""
Download and prepare training datasets for translation model.

This script downloads parallel corpora from Hugging Face datasets
and saves them in a clean JSON format for training.

Downloads:
- train-en-hi.json (1.6M+ sentence pairs)
- validation-en-hi.json (520 pairs)
- test-en-hi.json (2507 pairs)

Usage:
    python scripts/download_dataset.py
    python scripts/download_dataset.py --split validation  # Download specific split
    python scripts/download_dataset.py --all               # Force re-download all
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def download_en_hi_dataset(splits=None, force=False):
    """Download English-Hindi parallel corpus from IITB.
    
    Args:
        splits: List of splits to download. None means all ['train', 'validation', 'test']
        force: If True, re-download even if files exist
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install the datasets library:")
        print("  pip install datasets")
        return False

    if splits is None:
        splits = ["train", "validation", "test"]
    
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check which splits need downloading
    splits_to_download = []
    for split in splits:
        output_path = output_dir / f"{split}-en-hi.json"
        if force or not output_path.exists():
            splits_to_download.append(split)
        else:
            print(f"✓ {output_path} already exists (use --all to force re-download)")
    
    if not splits_to_download:
        print("\nAll files already exist!")
        return True
    
    print(f"\nDownloading English-Hindi dataset (IITB)...")
    print(f"Splits to download: {splits_to_download}")
    
    ds = load_dataset("cfilt/iitb-english-hindi")
    
    for split in splits_to_download:
        print(f"\nProcessing {split} split...")
        
        data = []
        for item in ds[split]:
            data.append({
                "source": item["translation"]["en"].strip(),
                "target": item["translation"]["hi"].strip()
            })
        
        output_path = output_dir / f"{split}-en-hi.json"
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Saved {len(data):,} samples to {output_path}")
    
    return True


def verify_dataset():
    """Verify all dataset files exist and are valid JSON."""
    output_dir = Path("data/raw")
    files = ["train-en-hi.json", "validation-en-hi.json", "test-en-hi.json"]
    
    print("\n--- Dataset Verification ---")
    all_valid = True
    
    for filename in files:
        filepath = output_dir / filename
        if not filepath.exists():
            print(f"❌ {filename}: NOT FOUND")
            all_valid = False
            continue
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"✅ {filename}: {len(data):,} samples")
        except json.JSONDecodeError as e:
            print(f"❌ {filename}: JSON ERROR at line {e.lineno}")
            all_valid = False
    
    return all_valid


def create_spm_corpus():
    """Create combined corpus for SentencePiece tokenizer training."""
    output_dir = Path("data/raw")
    train_file = output_dir / "train-en-hi.json"
    corpus_file = output_dir / "spm_corpus.txt"
    
    if not train_file.exists():
        print("❌ train-en-hi.json not found. Run download first.")
        return False
    
    print(f"\nCreating SentencePiece corpus from training data...")
    
    with open(train_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    with open(corpus_file, "w", encoding="utf-8") as f:
        for item in data:
            f.write(item["source"] + "\n")
            f.write(item["target"] + "\n")
    
    print(f"✅ Created {corpus_file} ({len(data) * 2:,} lines)")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download NMT training datasets")
    parser.add_argument("--split", type=str, choices=["train", "validation", "test"],
                       help="Download specific split only")
    parser.add_argument("--all", action="store_true",
                       help="Force re-download all splits")
    parser.add_argument("--verify", action="store_true",
                       help="Only verify existing files")
    parser.add_argument("--corpus", action="store_true",
                       help="Also create SentencePiece corpus file")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("NMT Dataset Downloader (IITB English-Hindi)")
    print("=" * 50)
    
    if args.verify:
        verify_dataset()
    else:
        splits = [args.split] if args.split else None
        success = download_en_hi_dataset(splits=splits, force=args.all)
        
        if success:
            verify_dataset()
            
            if args.corpus:
                create_spm_corpus()
    
    print("\nDone!")

