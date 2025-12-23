#!/usr/bin/env python3
"""
CLI Translation Tool for NMT.

Translate text using a trained model.

Usage:
    python scripts/translate.py --checkpoint best.pt --text "Hello, how are you?"
    python scripts/translate.py --checkpoint best.pt --file input.txt --output translations.txt
    python scripts/translate.py --checkpoint best.pt --interactive
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.nmt.tokenizer import Tokenizer
from src.nmt.model.transformer import Transformer
from src.nmt.inference import NMTTranslator


def parse_args():
    parser = argparse.ArgumentParser(description="Translate text with NMT model")
    
    # Model
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str,
                       default="models/translation/nmt_spm.model",
                       help="Path to tokenizer model")
    
    # Input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", type=str,
                            help="Text to translate")
    input_group.add_argument("--file", type=str,
                            help="File with texts to translate (one per line)")
    input_group.add_argument("--interactive", action="store_true",
                            help="Interactive translation mode")
    
    # Output
    parser.add_argument("--output", type=str, default=None,
                       help="Output file for translations")
    
    # Decoding
    parser.add_argument("--beam-size", type=int, default=4,
                       help="Beam size (0 for greedy)")
    parser.add_argument("--max-length", type=int, default=256,
                       help="Maximum output length")
    
    # Languages
    parser.add_argument("--source-lang", type=str, default="en",
                       help="Source language code")
    parser.add_argument("--target-lang", type=str, default="hi",
                       help="Target language code")
    
    # Device
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cuda, cpu)")
    
    return parser.parse_args()


def load_translator(args):
    """Load model and create translator."""
    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load tokenizer
    tokenizer = Tokenizer(
        model_path=args.tokenizer,
        language_tags=[f"<{args.source_lang}>", f"<{args.target_lang}>"]
    )
    
    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
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
        beam_size=args.beam_size,
        max_length=args.max_length
    )
    
    return translator, device


def translate_text(args):
    """Translate a single text."""
    translator, _ = load_translator(args)
    
    result = translator.translate(
        args.text,
        source_lang=f"<{args.source_lang}>",
        target_lang=f"<{args.target_lang}>"
    )
    
    print(f"\nSource ({args.source_lang}): {args.text}")
    print(f"Target ({args.target_lang}): {result}")


def translate_file(args):
    """Translate texts from a file."""
    translator, _ = load_translator(args)
    
    # Read input
    with open(args.file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    print(f"Translating {len(texts)} lines...")
    
    # Translate
    translations = translator.translate_batch(
        texts,
        source_lang=f"<{args.source_lang}>",
        target_lang=f"<{args.target_lang}>"
    )
    
    # Output
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            for t in translations:
                f.write(t + '\n')
        print(f"Translations saved to {args.output}")
    else:
        for src, tgt in zip(texts, translations):
            print(f"[SRC] {src}")
            print(f"[TGT] {tgt}")
            print()


def interactive_mode(args):
    """Interactive translation."""
    translator, device = load_translator(args)
    
    print("\n" + "=" * 50)
    print("Interactive Translation")
    print(f"  {args.source_lang} → {args.target_lang}")
    print(f"  Beam size: {args.beam_size}")
    print(f"  Device: {device}")
    print("=" * 50)
    print("Enter text to translate. Type 'quit' to exit.\n")
    
    while True:
        try:
            text = input(f"[{args.source_lang}] > ").strip()
            
            if not text:
                continue
            if text.lower() in ('quit', 'exit', 'q'):
                print("Goodbye!")
                break
            
            result = translator.translate(
                text,
                source_lang=f"<{args.source_lang}>",
                target_lang=f"<{args.target_lang}>"
            )
            
            print(f"[{args.target_lang}] > {result}\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    args = parse_args()
    
    if args.text:
        translate_text(args)
    elif args.file:
        translate_file(args)
    elif args.interactive:
        interactive_mode(args)


if __name__ == "__main__":
    main()
