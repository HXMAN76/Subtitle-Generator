"""
Download and prepare Samanantar dataset for translation model training.

Downloads parallel corpora from AI4Bharat's Samanantar dataset (HuggingFace)
and saves them in JSONL format for training.

Samanantar is the largest publicly available parallel corpora collection
for Indic languages with 49.6M English-Indic sentence pairs.

Supported languages: as, bn, gu, hi, kn, ml, mr, or, pa, ta, te

Usage:
    # Download Hindi dataset (default)
    python scripts/download_dataset.py
    
    # Download specific language
    python scripts/download_dataset.py --lang ta
    
    # Download multiple languages
    python scripts/download_dataset.py --lang hi ta te
    
    # Download all languages
    python scripts/download_dataset.py --all-langs
    
    # Verify existing files
    python scripts/download_dataset.py --verify
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nmt.languages import (
    SUPPORTED_LANGUAGES,
    LANGUAGE_SIZES,
    SOURCE_LANGUAGE,
    DEFAULT_TARGET_LANGUAGE,
    get_all_language_tags
)


def download_samanantar(
    languages: List[str],
    output_dir: Path,
    force: bool = False,
    max_samples: Optional[int] = None
) -> bool:
    """Download Samanantar dataset for specified languages.
    
    Args:
        languages: List of target language codes (e.g., ['hi', 'ta']).
        output_dir: Directory to save downloaded files.
        force: If True, re-download even if files exist.
        max_samples: Maximum number of samples per language (for testing).
    
    Returns:
        True if all downloads succeeded.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install the datasets library:")
        print("  pip install datasets")
        return False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    success = True
    
    for lang in languages:
        if lang not in SUPPORTED_LANGUAGES:
            print(f"❌ Unsupported language: {lang}")
            print(f"   Supported: {', '.join(sorted(SUPPORTED_LANGUAGES.keys()))}")
            success = False
            continue
        
        train_path = output_dir / f"train-en-{lang}.jsonl"
        
        if not force and train_path.exists():
            print(f"✓ {train_path.name} already exists (use --force to re-download)")
            continue
        
        lang_name = SUPPORTED_LANGUAGES[lang]
        size = LANGUAGE_SIZES.get(lang, "?")
        print(f"\n{'='*50}")
        print(f"Downloading: English → {lang_name} ({lang})")
        print(f"Approximate size: {size} sentence pairs")
        print(f"{'='*50}")
        
        try:
            # Load dataset for this language pair
            print(f"Loading from HuggingFace: ai4bharat/samanantar ({lang})...")
            ds = load_dataset(
                "ai4bharat/samanantar",
                lang,
                split="train",
                trust_remote_code=True
            )
            
            # Write to JSONL file
            print(f"Writing to {train_path}...")
            count = 0
            
            with open(train_path, "w", encoding="utf-8") as f:
                for item in ds:
                    if max_samples and count >= max_samples:
                        break
                    
                    # Samanantar format: src (English), tgt (target language)
                    record = {
                        "source": item["src"].strip(),
                        "target": item["tgt"].strip()
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
                    
                    if count % 100000 == 0:
                        print(f"  Processed {count:,} samples...")
            
            print(f"✅ Saved {count:,} samples to {train_path}")
            
        except Exception as e:
            print(f"❌ Error downloading {lang}: {e}")
            success = False
    
    return success


def verify_datasets(output_dir: Path) -> bool:
    """Verify all dataset files exist and are valid.
    
    Args:
        output_dir: Directory containing dataset files.
    
    Returns:
        True if all files are valid.
    """
    print("\n--- Dataset Verification ---")
    
    found_any = False
    all_valid = True
    
    for lang in sorted(SUPPORTED_LANGUAGES.keys()):
        train_path = output_dir / f"train-en-{lang}.jsonl"
        
        if not train_path.exists():
            continue
        
        found_any = True
        
        try:
            # Count lines and validate JSON
            count = 0
            with open(train_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        json.loads(line)  # Validate JSON
                        count += 1
            
            lang_name = SUPPORTED_LANGUAGES[lang]
            print(f"✅ {train_path.name}: {count:,} samples ({lang_name})")
            
        except json.JSONDecodeError as e:
            print(f"❌ {train_path.name}: JSON ERROR - {e}")
            all_valid = False
        except Exception as e:
            print(f"❌ {train_path.name}: ERROR - {e}")
            all_valid = False
    
    if not found_any:
        print("No dataset files found in", output_dir)
        print("Run: python scripts/download_dataset.py --lang hi")
        return False
    
    return all_valid


def create_combined_corpus(output_dir: Path, languages: List[str]) -> bool:
    """Create combined corpus for SentencePiece tokenizer training.
    
    Args:
        output_dir: Directory containing dataset files.
        languages: List of language codes to include.
    
    Returns:
        True if corpus was created successfully.
    """
    corpus_path = output_dir / "spm_corpus_multilang.txt"
    print(f"\nCreating combined corpus for tokenizer training...")
    
    total_lines = 0
    
    with open(corpus_path, "w", encoding="utf-8") as out_f:
        for lang in languages:
            train_path = output_dir / f"train-en-{lang}.jsonl"
            
            if not train_path.exists():
                print(f"  ⚠️ Skipping {lang}: file not found")
                continue
            
            print(f"  Processing {lang}...")
            count = 0
            
            with open(train_path, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    if line.strip():
                        record = json.loads(line)
                        # Write both source and target
                        out_f.write(record["source"] + "\n")
                        out_f.write(record["target"] + "\n")
                        count += 2
            
            total_lines += count
            print(f"    Added {count:,} lines from {lang}")
    
    print(f"\n✅ Created {corpus_path}")
    print(f"   Total lines: {total_lines:,}")
    
    return True


def create_validation_split(
    output_dir: Path,
    lang: str,
    val_size: int = 2000
) -> bool:
    """Create validation split from training data.
    
    Args:
        output_dir: Directory containing dataset files.
        lang: Language code.
        val_size: Number of samples for validation.
    
    Returns:
        True if split was created successfully.
    """
    train_path = output_dir / f"train-en-{lang}.jsonl"
    val_path = output_dir / f"validation-en-{lang}.jsonl"
    
    if not train_path.exists():
        print(f"❌ Training file not found: {train_path}")
        return False
    
    if val_path.exists():
        print(f"✓ Validation file already exists: {val_path}")
        return True
    
    print(f"Creating validation split for {lang} ({val_size} samples)...")
    
    # Read all lines, take last val_size for validation
    lines = []
    with open(train_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    if len(lines) <= val_size:
        print(f"  ⚠️ Not enough data for split (only {len(lines)} samples)")
        return False
    
    # Use last val_size as validation
    val_lines = lines[-val_size:]
    train_lines = lines[:-val_size]
    
    # Write validation file
    with open(val_path, "w", encoding="utf-8") as f:
        f.writelines(val_lines)
    
    # Rewrite training file without validation samples
    with open(train_path, "w", encoding="utf-8") as f:
        f.writelines(train_lines)
    
    print(f"✅ Created {val_path.name} ({val_size} samples)"ource /dist_home/nooglers/nooglers/Roshan/new/bin/activate
          
    print(f"   Training samples remaining: {len(train_lines):,}")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Samanantar dataset for NMT training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported Languages:
  as - Assamese    bn - Bengali     gu - Gujarati
  hi - Hindi       kn - Kannada     ml - Malayalam
  mr - Marathi     or - Odia        pa - Punjabi
  ta - Tamil       te - Telugu

Examples:
  python scripts/download_dataset.py --lang hi         # Download Hindi
  python scripts/download_dataset.py --lang hi ta te   # Download multiple
  python scripts/download_dataset.py --all-langs       # Download all
"""
    )
    
    parser.add_argument(
        "--lang", "-l",
        nargs="+",
        default=[DEFAULT_TARGET_LANGUAGE],
        help=f"Target language(s) to download (default: {DEFAULT_TARGET_LANGUAGE})"
    )
    parser.add_argument(
        "--all-langs",
        action="store_true",
        help="Download all supported languages"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="data/raw",
        help="Output directory (default: data/raw)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify existing files"
    )
    parser.add_argument(
        "--create-corpus",
        action="store_true",
        help="Create combined corpus for tokenizer training"
    )
    parser.add_argument(
        "--create-val-split",
        action="store_true",
        help="Create validation split from training data"
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=2000,
        help="Validation split size (default: 2000)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per language (for testing)"
    )
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    
    print("=" * 60)
    print("Samanantar Dataset Downloader")
    print("AI4Bharat English-Indic Parallel Corpus")
    print("=" * 60)
    
    # Determine which languages to download
    if args.all_langs:
        languages = list(SUPPORTED_LANGUAGES.keys())
        print(f"\nDownloading ALL {len(languages)} languages...")
    else:
        languages = args.lang
        print(f"\nLanguages: {', '.join(languages)}")
    
    if args.verify:
        verify_datasets(output_dir)
    else:
        success = download_samanantar(
            languages=languages,
            output_dir=output_dir,
            force=args.force,
            max_samples=args.max_samples
        )
        
        if success:
            verify_datasets(output_dir)
            
            if args.create_val_split:
                for lang in languages:
                    create_validation_split(output_dir, lang, args.val_size)
            
            if args.create_corpus:
                create_combined_corpus(output_dir, languages)
    
    print("\nDone!")
