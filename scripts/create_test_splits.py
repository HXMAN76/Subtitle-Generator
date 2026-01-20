#!/usr/bin/env python3
"""
Create Test/Validation Splits for All Languages.

Extracts test and validation splits from Samanantar training data
for all supported languages.

Usage:
    # Create test splits for all available languages
    python scripts/create_test_splits.py
    
    # Create splits for specific languages
    python scripts/create_test_splits.py --lang hi ta bn
    
    # Specify custom split sizes
    python scripts/create_test_splits.py --test-size 1000 --val-size 2000
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nmt.languages import SUPPORTED_LANGUAGES, get_language_name


def create_test_val_splits(
    lang: str,
    data_dir: Path,
    test_size: int = 1000,
    val_size: int = 2000,
    force: bool = False
) -> bool:
    """Create test and validation splits from training data.
    
    Args:
        lang: Language code.
        data_dir: Directory containing dataset files.
        test_size: Number of samples for test set.
        val_size: Number of samples for validation set.
        force: If True, overwrite existing splits.
        
    Returns:
        True if splits were created successfully.
    """
    # File paths (JSONL format from Samanantar)
    train_jsonl = data_dir / f"train-en-{lang}.jsonl"
    
    # Output paths (JSON format for compatibility)
    test_json = data_dir / f"test-en-{lang}.json"
    val_json = data_dir / f"validation-en-{lang}.json"
    train_json = data_dir / f"train-en-{lang}.json"
    
    # Check if source file exists
    if not train_jsonl.exists():
        print(f"  ❌ Training file not found: {train_jsonl.name}")
        return False
    
    # Check if splits already exist
    if test_json.exists() and val_json.exists() and not force:
        print(f"  ✓ Splits already exist for {lang} (use --force to recreate)")
        return True
    
    lang_name = get_language_name(lang)
    print(f"\nProcessing: {lang_name} ({lang})")
    print(f"  Reading from: {train_jsonl.name}")
    
    # Read all samples
    samples = []
    try:
        with open(train_jsonl, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        sample = json.loads(line)
                        samples.append(sample)
                    except json.JSONDecodeError as e:
                        print(f"  ⚠️ Skipping line {line_num}: {e}")
                
                # Progress indicator
                if line_num % 100000 == 0:
                    print(f"    Read {line_num:,} lines...", end='\r')
    
    except Exception as e:
        print(f"  ❌ Error reading file: {e}")
        return False
    
    print(f"  Total samples: {len(samples):,}")
    
    # Check if we have enough data
    total_needed = test_size + val_size
    if len(samples) < total_needed:
        print(f"  ⚠️ Not enough data (need {total_needed:,}, have {len(samples):,})")
        print(f"     Using available data with adjusted sizes...")
        test_size = min(test_size, len(samples) // 3)
        val_size = min(val_size, len(samples) // 3)
    
    # Split data: use last samples for test/validation
    test_samples = samples[-(test_size + val_size):-val_size] if val_size > 0 else samples[-test_size:]
    val_samples = samples[-val_size:] if val_size > 0 else []
    train_samples = samples[:-(test_size + val_size)]
    
    # Save test split as JSON
    print(f"  Writing test split ({len(test_samples):,} samples)...")
    with open(test_json, 'w', encoding='utf-8') as f:
        json.dump(test_samples, f, ensure_ascii=False, indent=2)
    print(f"  ✅ Created: {test_json.name}")
    
    # Save validation split as JSON
    if val_samples:
        print(f"  Writing validation split ({len(val_samples):,} samples)...")
        with open(val_json, 'w', encoding='utf-8') as f:
            json.dump(val_samples, f, ensure_ascii=False, indent=2)
        print(f"  ✅ Created: {val_json.name}")
    
    # Optionally save updated training split as JSON
    print(f"  Training samples remaining: {len(train_samples):,}")
    
    # Save train split as JSON (for compatibility)
    if not train_json.exists() or force:
        print(f"  Writing training split as JSON ({len(train_samples):,} samples)...")
        with open(train_json, 'w', encoding='utf-8') as f:
            json.dump(train_samples, f, ensure_ascii=False, indent=2)
        print(f"  ✅ Created: {train_json.name}")
    
    return True


def find_available_languages(data_dir: Path) -> List[str]:
    """Find languages that have training data available.
    
    Args:
        data_dir: Directory containing dataset files.
        
    Returns:
        List of language codes with available data.
    """
    available = []
    
    for lang in SUPPORTED_LANGUAGES.keys():
        train_file = data_dir / f"train-en-{lang}.jsonl"
        if train_file.exists():
            available.append(lang)
    
    return available


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create test/validation splits for all languages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create splits for all available languages
  python scripts/create_test_splits.py

  # Create splits for specific languages
  python scripts/create_test_splits.py --lang hi ta bn

  # Custom split sizes
  python scripts/create_test_splits.py --test-size 1000 --val-size 2000

  # Force recreation of existing splits
  python scripts/create_test_splits.py --force
        """
    )
    
    parser.add_argument(
        "--lang", "-l",
        nargs="+",
        default=None,
        help="Specific language(s) to process (default: all available)"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Data directory (default: data/raw)"
    )
    
    parser.add_argument(
        "--test-size",
        type=int,
        default=1000,
        help="Test set size (default: 1000)"
    )
    
    parser.add_argument(
        "--val-size",
        type=int,
        default=2000,
        help="Validation set size (default: 2000)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recreation of existing splits"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    
    print("=" * 60)
    print("Create Test/Validation Splits")
    print("=" * 60)
    
    # Find available languages
    available_langs = find_available_languages(data_dir)
    
    if not available_langs:
        print(f"\n❌ No training data found in {data_dir}")
        print("\nDownload data first:")
        print("  python scripts/download_dataset.py --all-langs")
        sys.exit(1)
    
    print(f"\nAvailable languages ({len(available_langs)}):")
    for lang in sorted(available_langs):
        print(f"  - {lang} ({get_language_name(lang)})")
    
    # Determine which languages to process
    if args.lang:
        languages = [l for l in args.lang if l in available_langs]
        missing = [l for l in args.lang if l not in available_langs]
        
        if missing:
            print(f"\n⚠️ Skipping unavailable languages: {', '.join(missing)}")
        
        if not languages:
            print("\n❌ None of the specified languages have data available")
            sys.exit(1)
    else:
        languages = available_langs
    
    print(f"\nConfiguration:")
    print(f"  Test size:       {args.test_size:,}")
    print(f"  Validation size: {args.val_size:,}")
    print(f"  Force recreate:  {args.force}")
    
    # Create splits for each language
    print(f"\nProcessing {len(languages)} language(s)...")
    print("=" * 60)
    
    successes = 0
    failures = 0
    
    for lang in sorted(languages):
        try:
            if create_test_val_splits(
                lang=lang,
                data_dir=data_dir,
                test_size=args.test_size,
                val_size=args.val_size,
                force=args.force
            ):
                successes += 1
            else:
                failures += 1
        except Exception as e:
            print(f"  ❌ Error processing {lang}: {e}")
            failures += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✅ Successful: {successes}/{len(languages)}")
    if failures > 0:
        print(f"❌ Failed: {failures}/{len(languages)}")
    
    print("\nTest files created:")
    for lang in sorted(languages):
        test_file = data_dir / f"test-en-{lang}.json"
        if test_file.exists():
            print(f"  ✓ {test_file}")


if __name__ == "__main__":
    main()
