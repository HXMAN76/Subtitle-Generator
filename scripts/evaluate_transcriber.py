#!/usr/bin/env python3
"""
Evaluation Script for Transcriber (ASR).

Evaluate transcription quality using industry-standard ASR metrics:
- WER (Word Error Rate)
- CER (Character Error Rate)
- SER (Sentence Error Rate)
- RTF (Real-Time Factor)

Usage:
    python scripts/evaluate_transcriber.py --test-data data/asr/test_transcriptions.json
    python scripts/evaluate_transcriber.py --test-data data/asr/test_transcriptions.json --output results/asr_report.json
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transcriber import Transcriber
from src.transcriber_evaluation import TranscriberEvaluator, ASRMetricsResult
import config


def load_test_data(test_file: str) -> list:
    """Load test data from JSON file.
    
    Expected format:
    [
        {
            "audio": "path/to/audio.wav",
            "reference": "ground truth transcription text",
            "duration": 5.2  # optional, in seconds
        }
    ]
    
    Args:
        test_file: Path to test data JSON file.
        
    Returns:
        List of test examples.
    """
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def save_report(result: ASRMetricsResult, output_path: str, test_file: str, model_size: str):
    """Save evaluation report as JSON.
    
    Args:
        result: ASRMetricsResult with metrics.
        output_path: Path to save report.
        test_file: Path to test data file.
        model_size: Whisper model size.
    """
    report = {
        "timestamp": datetime.now().isoformat(),
        "model": {
            "type": "faster-whisper",
            "size": model_size,
            "device": config.WHISPER_DEVICE
        },
        "test_data": test_file,
        "metrics": result.to_dict()
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\nReport saved to: {output_path}")


def print_detailed_results(result: ASRMetricsResult):
    """Print detailed evaluation results.
    
    Args:
        result: ASRMetricsResult with metrics.
    """
    print("\n" + "=" * 60)
    print("TRANSCRIBER (ASR) EVALUATION RESULTS")
    print("=" * 60)
    
    if result.wer is not None:
        print(f"\nWord Error Rate (WER):      {result.wer:.2%}")
        if result.total_words:
            print(f"  Substitutions:            {result.substitutions}/{result.total_words}")
            print(f"  Deletions:                {result.deletions}/{result.total_words}")
            print(f"  Insertions:               {result.insertions}/{result.total_words}")
    
    if result.cer is not None:
        print(f"\nCharacter Error Rate (CER): {result.cer:.2%}")
    
    if result.ser is not None:
        print(f"\nSentence Error Rate (SER):  {result.ser:.2%}")
    
    if result.rtf is not None:
        print(f"\nReal-Time Factor (RTF):     {result.rtf:.3f}x")
        if result.rtf < 1.0:
            print(f"  ✓ Faster than real-time ({1/result.rtf:.1f}x speedup)")
        else:
            print(f"  ⚠ Slower than real-time")
    
    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Transcriber (ASR) performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on test dataset
  python scripts/evaluate_transcriber.py --test-data data/asr/test_transcriptions.json

  # Use different model size
  python scripts/evaluate_transcriber.py --test-data data/asr/test.json --model-size large-v3

  # Save detailed report
  python scripts/evaluate_transcriber.py --test-data data/asr/test.json --output results/asr_report.json
        """
    )
    
    parser.add_argument("--test-data", type=str, required=True,
                       help="Path to test data JSON file")
    
    parser.add_argument("--model-size", type=str, default=None,
                       help="Whisper model size (tiny, base, small, medium, large-v3). Uses config default if not specified.")
    
    parser.add_argument("--output", type=str, default=None,
                       help="Path to save evaluation report")
    
    parser.add_argument("--normalize", action="store_true", default=True,
                       help="Normalize text before comparison (default: True)")
    parser.add_argument("--no-normalize", action="store_false", dest="normalize",
                       help="Don't normalize text before comparison")
    
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Print progress during evaluation (default: True)")
    parser.add_argument("--quiet", action="store_false", dest="verbose",
                       help="Suppress progress messages")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Transcriber (ASR) Evaluation")
    print("=" * 60)
    
    # Determine model size
    model_size = args.model_size or config.WHISPER_MODEL_SIZE
    print(f"Model: faster-whisper ({model_size})")
    print(f"Device: {config.WHISPER_DEVICE}")
    
    # Load test data
    print(f"\nLoading test data from: {args.test_data}")
    test_data = load_test_data(args.test_data)
    print(f"Loaded {len(test_data)} test examples")
    
    # Validate test data
    for i, item in enumerate(test_data):
        if 'audio' not in item or 'reference' not in item:
            print(f"Error: Test example {i} missing 'audio' or 'reference' field")
            sys.exit(1)
        
        # Check if audio file exists
        if not Path(item['audio']).exists():
            print(f"Warning: Audio file not found: {item['audio']}")
    
    # Initialize transcriber
    print(f"\nInitializing Transcriber...")
    transcriber = Transcriber(model_size=model_size)
    
    # Create evaluator
    evaluator = TranscriberEvaluator(
        transcriber=transcriber,
        normalize=args.normalize
    )
    
    # Evaluate
    print(f"\nEvaluating on {len(test_data)} examples...")
    result = evaluator.evaluate_dataset(
        test_data=test_data,
        verbose=args.verbose
    )
    
    # Print results
    print_detailed_results(result)
    
    # Save report if requested
    if args.output:
        save_report(result, args.output, args.test_data, model_size)


if __name__ == "__main__":
    main()
