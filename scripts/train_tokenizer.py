#!/usr/bin/env python3
"""
Train Per-Language Tokenizers for NMT.

Creates optimized SentencePiece tokenizers for each language pair (en-X).
Each tokenizer has full 32K vocabulary dedicated to its language pair,
providing better coverage for complex scripts like Dravidian languages.

Usage:
    # Train tokenizer for a specific language
    python scripts/train_tokenizer.py --target-lang ta
    
    # Train tokenizers for all languages
    python scripts/train_tokenizer.py --all
    
    # Custom vocabulary size for Dravidian languages
    python scripts/train_tokenizer.py --target-lang kn --vocab-size 48000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nmt.languages import SUPPORTED_LANGUAGES, get_language_name, DRAVIDIAN_LANGUAGES


def create_corpus_for_language(target_lang: str, data_dir: Path, output_path: Path,
                               max_samples: int = 2_000_000) -> bool:
    """Create a bilingual corpus for tokenizer training.
    
    Args:
        target_lang: Target language code.
        data_dir: Directory containing training data.
        output_path: Path to write corpus file.
        max_samples: Maximum number of sentence pairs to include.
        
    Returns:
        True if corpus was created successfully.
    """
    import json
    
    train_file = data_dir / f"train-en-{target_lang}.jsonl"
    if not train_file.exists():
        print(f"Training data not found: {train_file}")
        return False
    
    print(f"Creating corpus from {train_file}...")
    count = 0
    

    
    # Auto-detect keys from first line
    src_key = None
    tgt_key = None
    
    with open(output_path, 'w', encoding='utf-8') as out:
        with open(train_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if count >= max_samples:
                    break
                try:
                    item = json.loads(line.strip())
                    
                    # Detect keys on first line
                    if src_key is None:
                        keys = list(item.keys())
                        if 'src' in keys and 'tgt' in keys:
                            src_key, tgt_key = 'src', 'tgt'
                        elif 'source' in keys and 'target' in keys:
                            src_key, tgt_key = 'source', 'target'
                        elif 'en' in keys and target_lang in keys:
                            src_key, tgt_key = 'en', target_lang
                        elif 'english' in keys and get_language_name(target_lang).lower() in keys:
                            src_key, tgt_key = 'english', get_language_name(target_lang).lower()
                        else:
                            # Fallback/Debug
                            if i == 0:
                                print(f"⚠️ Warning: Could not auto-detect keys. Found: {keys}")
                                print(f"   Expected: ['src', 'tgt'] or ['source', 'target'] or ['en', '{target_lang}']")
                            src_key, tgt_key = 'src', 'tgt' # Default
                    
                    src = item.get(src_key, '').strip()
                    tgt = item.get(tgt_key, '').strip()
                    
                    if src and tgt:
                        out.write(src + '\n')
                        out.write(tgt + '\n')
                        count += 1
                except json.JSONDecodeError:
                    continue
    
    if count == 0:
        print(f"❌ Failed to extract any sentences from {train_file}")
        if src_key:
            print(f"   Using keys: src='{src_key}', tgt='{tgt_key}'")
    else:
        print(f"Created corpus with {count} sentence pairs ({count * 2} sentences)")
    
    return count > 0


def train_tokenizer_for_language(
    target_lang: str,
    data_dir: Path = Path("data/raw"),
    output_dir: Path = Path("models/translation"),
    vocab_size: int = None,
    model_type: str = None
):
    """Train a tokenizer for a specific language pair.
    
    Args:
        target_lang: Target language code.
        data_dir: Directory containing training data.
        output_dir: Directory to save tokenizer.
        vocab_size: Vocabulary size (auto-selected if None).
        model_type: "bpe" or "unigram" (auto-selected if None).
    """
    from src.nmt.tokenizer import Tokenizer
    
    lang_name = get_language_name(target_lang)
    print(f"\n{'='*60}")
    print(f"Training Tokenizer: English → {lang_name} ({target_lang})")
    print('='*60)
    
    # Auto-select parameters based on language family
    if vocab_size is None:
        vocab_size = 48000 if target_lang in DRAVIDIAN_LANGUAGES else 32000
    if model_type is None:
        model_type = "unigram" if target_lang in DRAVIDIAN_LANGUAGES else "bpe"
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Model type: {model_type}")
    print(f"Language family: {'Dravidian' if target_lang in DRAVIDIAN_LANGUAGES else 'Indo-Aryan'}")
    
    # Create output directory
    lang_output_dir = output_dir / target_lang
    lang_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temporary corpus file
    corpus_path = lang_output_dir / "tokenizer_corpus.txt"
    if not create_corpus_for_language(target_lang, data_dir, corpus_path):
        print(f"Failed to create corpus for {target_lang}")
        return False
    
    # Train tokenizer
    tokenizer_prefix = str(lang_output_dir / "tokenizer")
    language_tags = ["<en>", f"<{target_lang}>"]
    
    print(f"\nTraining SentencePiece tokenizer...")
    
    Tokenizer.train(
        corpus_path=str(corpus_path),
        model_prefix=tokenizer_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=0.9999,  # High coverage for Indic scripts
        language_tags=language_tags
    )
    
    # Clean up corpus file (optional, keep for debugging)
    # corpus_path.unlink()
    
    # Verify the tokenizer works
    tokenizer = Tokenizer(
        model_path=f"{tokenizer_prefix}.model",
        language_tags=language_tags
    )
    
    print(f"\n✓ Tokenizer saved: {tokenizer_prefix}.model")
    print(f"  Vocabulary size: {tokenizer.vocab_size}")
    
    # Test tokenization
    test_en = "Hello, how are you today?"
    ids_en = tokenizer.encode(test_en, add_lang_tag="<en>")
    print(f"\n  Test EN: '{test_en}' → {len(ids_en)} tokens")
    
    return True


def train_all_tokenizers(data_dir: Path, output_dir: Path, vocab_size: int = None):
    """Train tokenizers for all supported languages."""
    results = {}
    
    for lang in sorted(SUPPORTED_LANGUAGES.keys()):
        try:
            success = train_tokenizer_for_language(
                target_lang=lang,
                data_dir=data_dir,
                output_dir=output_dir,
                vocab_size=vocab_size
            )
            results[lang] = "✓ Success" if success else "✗ Failed"
        except Exception as e:
            results[lang] = f"✗ Error: {e}"
            print(f"Error training tokenizer for {lang}: {e}")
    
    # Print summary
    print("\n" + "="*60)
    print("TOKENIZER TRAINING SUMMARY")
    print("="*60)
    for lang, status in results.items():
        lang_name = get_language_name(lang)
        print(f"  {lang} ({lang_name:12s}): {status}")
    
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train per-language tokenizers for NMT",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--target-lang", "-t", type=str,
                       choices=list(SUPPORTED_LANGUAGES.keys()),
                       help="Target language code")
    
    parser.add_argument("--all", action="store_true",
                       help="Train tokenizers for all languages")
    
    parser.add_argument("--vocab-size", type=int, default=None,
                       help="Vocabulary size (default: 32K for Indo-Aryan, 48K for Dravidian)")
    
    parser.add_argument("--model-type", type=str, choices=["bpe", "unigram"],
                       default=None,
                       help="SentencePiece model type (default: bpe for Indo-Aryan, unigram for Dravidian)")
    
    parser.add_argument("--data-dir", type=str, default="data/raw",
                       help="Directory containing training data")
    
    parser.add_argument("--output-dir", type=str, default="models/translation",
                       help="Output directory for tokenizers")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if not args.all and not args.target_lang:
        print("Error: Specify either --target-lang or --all")
        print("Use -h for help")
        return
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    if args.all:
        train_all_tokenizers(data_dir, output_dir, args.vocab_size)
    else:
        train_tokenizer_for_language(
            target_lang=args.target_lang,
            data_dir=data_dir,
            output_dir=output_dir,
            vocab_size=args.vocab_size,
            model_type=args.model_type
        )


if __name__ == "__main__":
    main()
