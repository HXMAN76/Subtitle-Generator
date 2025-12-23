#!/usr/bin/env python3
"""
Evaluation Script for NMT Transformer.

Evaluate trained models on test sets and compute metrics.

Usage:
    python scripts/evaluate_nmt.py --checkpoint best.pt --test-data test-en-hi.json
    python scripts/evaluate_nmt.py --checkpoint best.pt --samples 10
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.nmt.tokenizer import Tokenizer
from src.nmt.model.transformer import Transformer
from src.nmt.inference import NMTTranslator
from src.nmt.evaluation import Evaluator, compute_bleu


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate NMT model")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str,
                       default="models/translation/nmt_spm.model",
                       help="Path to tokenizer model")
    parser.add_argument("--test-data", type=str,
                       default="data/raw/test-en-hi.json",
                       help="Path to test data JSON file")
    
    parser.add_argument("--output", type=str, default=None,
                       help="Path to save translations")
    parser.add_argument("--samples", type=int, default=0,
                       help="Number of sample translations to show")
    
    parser.add_argument("--beam-size", type=int, default=4,
                       help="Beam size for decoding")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for translation")
    
    parser.add_argument("--comet", action="store_true",
                       help="Compute COMET score (slower)")
    
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cuda, cpu)")
    
    parser.add_argument("--source-lang", type=str, default="en",
                       help="Source language code")
    parser.add_argument("--target-lang", type=str, default="hi",
                       help="Target language code")
    
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
    tokenizer = Tokenizer(
        model_path=args.tokenizer,
        language_tags=[f"<{args.source_lang}>", f"<{args.target_lang}>"]
    )
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    config = checkpoint.get('config', {})
    config['vocab_size'] = tokenizer.vocab_size
    
    model = Transformer.from_config(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded: {model.count_parameters_readable()} parameters")
    
    # Create translator
    translator = NMTTranslator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        beam_size=args.beam_size
    )
    
    # Create evaluator
    evaluator = Evaluator(
        translator=translator,
        source_lang=f"<{args.source_lang}>",
        target_lang=f"<{args.target_lang}>"
    )
    
    # Show sample translations if requested
    if args.samples > 0:
        print(f"\n--- Sample Translations ({args.samples}) ---\n")
        samples = evaluator.sample_translations(
            args.test_data,
            n_samples=args.samples
        )
        
        for i, sample in enumerate(samples, 1):
            print(f"[{i}]")
            print(f"  Source:     {sample['source'][:100]}...")
            print(f"  Reference:  {sample['reference'][:100]}...")
            print(f"  Hypothesis: {sample['hypothesis'][:100]}...")
            print()
    
    # Evaluate
    print(f"\n--- Evaluation on {args.test_data} ---")
    
    result = evaluator.evaluate_file(
        test_file=args.test_data,
        batch_size=args.batch_size,
        output_file=args.output,
        include_comet=args.comet
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
