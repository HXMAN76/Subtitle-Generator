#!/usr/bin/env python3
"""
Evaluation Script for NMT Transformer.

Evaluate trained models on test sets and compute metrics.
Supports multi-language evaluation with automatic model discovery.

Usage:
    # Single language evaluation
    python scripts/evaluate_nmt.py --checkpoint models/translation/hi/best.pt --target-lang hi
    
    # Evaluate all available language models
    python scripts/evaluate_nmt.py --all-languages
    
    # Show sample translations
    python scripts/evaluate_nmt.py --checkpoint best.pt --samples 10
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.nmt.tokenizer import Tokenizer
from src.nmt.model.transformer import Transformer
from src.nmt.inference import NMTTranslator
from src.nmt.evaluation import Evaluator, MetricsResult
from src.nmt.languages import SUPPORTED_LANGUAGES, get_language_name


# Default paths
MODELS_DIR = Path("models/translation")
DATA_DIR = Path("data/raw")
TOKENIZER_PATH = MODELS_DIR / "nmt_spm.model"


def discover_language_models() -> Dict[str, Path]:
    """Auto-discover available language models.
    
    Scans models/translation/{lang}/ directories for best.pt checkpoints.
    
    Returns:
        Dict mapping language codes to checkpoint paths.
    """
    available = {}
    
    for lang_code in SUPPORTED_LANGUAGES.keys():
        lang_dir = MODELS_DIR / lang_code
        checkpoint = lang_dir / "best.pt"
        
        if checkpoint.exists():
            available[lang_code] = checkpoint
    
    return available


def find_test_data(lang_code: str) -> Optional[Path]:
    """Find test data file for a language.
    
    Looks for test-en-{lang}.json in data/raw/
    
    Args:
        lang_code: Target language code.
        
    Returns:
        Path to test file if found, None otherwise.
    """
    test_file = DATA_DIR / f"test-en-{lang_code}.json"
    return test_file if test_file.exists() else None


def evaluate_single_language(
    checkpoint_path: Path,
    target_lang: str,
    test_file: Path,
    tokenizer: Tokenizer,
    device: torch.device,
    beam_size: int = 4,
    batch_size: int = 32,
    include_comet: bool = False,
    samples: int = 0,
    output_file: Optional[str] = None
) -> MetricsResult:
    """Evaluate a single language model.
    
    Args:
        checkpoint_path: Path to model checkpoint.
        target_lang: Target language code.
        test_file: Path to test data.
        tokenizer: Shared tokenizer.
        device: Compute device.
        beam_size: Beam size for decoding.
        batch_size: Batch size for translation.
        include_comet: Whether to compute COMET.
        samples: Number of sample translations to show.
        output_file: Optional path to save translations.
        
    Returns:
        MetricsResult with evaluation metrics.
    """
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint.get('config', {})
    config['vocab_size'] = tokenizer.vocab_size
    
    model = Transformer.from_config(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Create translator
    translator = NMTTranslator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        beam_size=beam_size
    )
    
    # Create evaluator
    evaluator = Evaluator(
        translator=translator,
        source_lang="<en>",
        target_lang=f"<{target_lang}>"
    )
    
    # Show sample translations if requested
    if samples > 0:
        print(f"\n--- Sample Translations ({samples}) ---\n")
        sample_results = evaluator.sample_translations(
            str(test_file),
            n_samples=samples
        )
        
        for i, sample in enumerate(sample_results, 1):
            print(f"[{i}]")
            print(f"  Source:     {sample['source'][:100]}...")
            print(f"  Reference:  {sample['reference'][:100]}...")
            print(f"  Hypothesis: {sample['hypothesis'][:100]}...")
            print()
    
    # Evaluate
    result = evaluator.evaluate_file(
        test_file=str(test_file),
        batch_size=batch_size,
        output_file=output_file,
        include_comet=include_comet
    )
    
    return result


def evaluate_all_languages(
    tokenizer: Tokenizer,
    device: torch.device,
    beam_size: int = 4,
    batch_size: int = 32,
    include_comet: bool = False,
    output_dir: Optional[str] = None
) -> Dict[str, MetricsResult]:
    """Evaluate all available language models.
    
    Args:
        tokenizer: Shared tokenizer.
        device: Compute device.
        beam_size: Beam size for decoding.
        batch_size: Batch size for translation.
        include_comet: Whether to compute COMET.
        output_dir: Optional directory to save individual results.
        
    Returns:
        Dict mapping language codes to MetricsResult.
    """
    available_models = discover_language_models()
    
    if not available_models:
        print("No language models found in models/translation/")
        return {}
    
    print(f"\nFound {len(available_models)} language model(s):")
    for lang, path in sorted(available_models.items()):
        print(f"  - {lang} ({get_language_name(lang)}): {path}")
    
    results = {}
    
    for lang_code, checkpoint_path in sorted(available_models.items()):
        test_file = find_test_data(lang_code)
        
        if not test_file:
            print(f"\n⚠ Skipping {lang_code}: No test data found (expected: {DATA_DIR}/test-en-{lang_code}.json)")
            continue
        
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {get_language_name(lang_code)} ({lang_code})")
        print(f"{'=' * 60}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Test data:  {test_file}")
        
        output_file = None
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            output_file = str(Path(output_dir) / f"translations-en-{lang_code}.json")
        
        try:
            result = evaluate_single_language(
                checkpoint_path=checkpoint_path,
                target_lang=lang_code,
                test_file=test_file,
                tokenizer=tokenizer,
                device=device,
                beam_size=beam_size,
                batch_size=batch_size,
                include_comet=include_comet,
                output_file=output_file
            )
            results[lang_code] = result
            
        except Exception as e:
            print(f"✗ Error evaluating {lang_code}: {e}")
            continue
    
    return results


def print_summary_table(results: Dict[str, MetricsResult]):
    """Print a summary table of all language results.
    
    Args:
        results: Dict mapping language codes to MetricsResult.
    """
    if not results:
        return
    
    print("\n" + "=" * 70)
    print("MULTI-LANGUAGE EVALUATION SUMMARY")
    print("=" * 70)
    
    # Header
    print(f"{'Language':<15} {'Code':<6} {'BLEU':<10} {'METEOR':<10} {'COMET':<10}")
    print("-" * 70)
    
    # Results
    for lang_code, result in sorted(results.items()):
        lang_name = get_language_name(lang_code)
        bleu = f"{result.bleu:.2f}" if result.bleu is not None else "N/A"
        meteor = f"{result.meteor:.4f}" if result.meteor is not None else "N/A"
        comet = f"{result.comet:.4f}" if result.comet is not None else "N/A"
        
        print(f"{lang_name:<15} {lang_code:<6} {bleu:<10} {meteor:<10} {comet:<10}")
    
    # Averages
    print("-" * 70)
    bleu_scores = [r.bleu for r in results.values() if r.bleu is not None]
    meteor_scores = [r.meteor for r in results.values() if r.meteor is not None]
    comet_scores = [r.comet for r in results.values() if r.comet is not None]
    
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0
    avg_comet = sum(comet_scores) / len(comet_scores) if comet_scores else 0
    
    print(f"{'AVERAGE':<15} {'':<6} {avg_bleu:<10.2f} {avg_meteor:<10.4f} {avg_comet:<10.4f}")
    print("=" * 70)


def save_report(results: Dict[str, MetricsResult], output_path: str):
    """Save evaluation report as JSON.
    
    Args:
        results: Dict mapping language codes to MetricsResult.
        output_path: Path to save report.
    """
    report = {
        "timestamp": datetime.now().isoformat(),
        "languages": len(results),
        "results": {
            lang: {
                "language_name": get_language_name(lang),
                **result.to_dict()
            }
            for lang, result in results.items()
        },
        "averages": {}
    }
    
    # Compute averages
    bleu_scores = [r.bleu for r in results.values() if r.bleu is not None]
    meteor_scores = [r.meteor for r in results.values() if r.meteor is not None]
    comet_scores = [r.comet for r in results.values() if r.comet is not None]
    
    if bleu_scores:
        report["averages"]["bleu"] = sum(bleu_scores) / len(bleu_scores)
    if meteor_scores:
        report["averages"]["meteor"] = sum(meteor_scores) / len(meteor_scores)
    if comet_scores:
        report["averages"]["comet"] = sum(comet_scores) / len(comet_scores)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\nReport saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate NMT model(s)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate single language
  python scripts/evaluate_nmt.py --checkpoint models/translation/hi/best.pt --target-lang hi

  # Evaluate all available languages
  python scripts/evaluate_nmt.py --all-languages

  # Evaluate all with COMET and save report
  python scripts/evaluate_nmt.py --all-languages --comet --output results/nmt_report.json
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--checkpoint", type=str,
                           help="Path to model checkpoint (single language mode)")
    mode_group.add_argument("--all-languages", action="store_true",
                           help="Evaluate all available language models")
    
    # Paths
    parser.add_argument("--tokenizer", type=str,
                       default=str(TOKENIZER_PATH),
                       help="Path to tokenizer model")
    parser.add_argument("--test-data", type=str, default=None,
                       help="Path to test data (auto-detected for --all-languages)")
    parser.add_argument("--output", type=str, default=None,
                       help="Path to save report/translations")
    
    # Language (for single mode)
    parser.add_argument("--source-lang", type=str, default="en",
                       help="Source language code")
    parser.add_argument("--target-lang", type=str, default="hi",
                       help="Target language code")
    
    # Decoding
    parser.add_argument("--beam-size", type=int, default=4,
                       help="Beam size for decoding")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for translation")
    
    # Metrics
    parser.add_argument("--comet", action="store_true",
                       help="Compute COMET score (slower, requires GPU)")
    
    # Display
    parser.add_argument("--samples", type=int, default=0,
                       help="Number of sample translations to show")
    
    # Hardware
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cuda, cpu)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("NMT Evaluation")
    print("=" * 60)
    
    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load tokenizer
    print(f"\nLoading tokenizer from {args.tokenizer}")
    
    # Get all language tags for multi-language support
    from src.nmt.languages import get_all_language_tags
    all_lang_tags = get_all_language_tags()
    
    tokenizer = Tokenizer(
        model_path=args.tokenizer,
        language_tags=all_lang_tags
    )
    
    if args.all_languages:
        # Multi-language evaluation
        results = evaluate_all_languages(
            tokenizer=tokenizer,
            device=device,
            beam_size=args.beam_size,
            batch_size=args.batch_size,
            include_comet=args.comet,
            output_dir=args.output
        )
        
        # Print summary
        print_summary_table(results)
        
        # Save report
        if args.output:
            report_path = args.output
            if Path(args.output).is_dir():
                report_path = str(Path(args.output) / "nmt_evaluation_report.json")
            save_report(results, report_path)
    
    else:
        # Single language evaluation
        checkpoint_path = Path(args.checkpoint)
        
        # Find test data
        if args.test_data:
            test_file = Path(args.test_data)
        else:
            test_file = find_test_data(args.target_lang)
            if not test_file:
                test_file = DATA_DIR / f"test-en-{args.target_lang}.json"
        
        if not test_file.exists():
            print(f"Error: Test data not found: {test_file}")
            sys.exit(1)
        
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Test data:  {test_file}")
        print(f"Language:   en → {args.target_lang} ({get_language_name(args.target_lang)})")
        
        result = evaluate_single_language(
            checkpoint_path=checkpoint_path,
            target_lang=args.target_lang,
            test_file=test_file,
            tokenizer=tokenizer,
            device=device,
            beam_size=args.beam_size,
            batch_size=args.batch_size,
            include_comet=args.comet,
            samples=args.samples,
            output_file=args.output
        )
        
        print("\n" + "=" * 40)
        print("RESULTS")
        print("=" * 40)
        print(f"BLEU:   {result.bleu:.2f}" if result.bleu else "BLEU:   N/A")
        print(f"METEOR: {result.meteor:.4f}" if result.meteor else "METEOR: N/A")
        print(f"COMET:  {result.comet:.4f}" if result.comet else "COMET:  N/A")
        
        if result.bleu_signature:
            print(f"\nBLEU Signature: {result.bleu_signature}")


if __name__ == "__main__":
    main()
