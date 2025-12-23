#!/usr/bin/env python3
"""
Training Script for NMT Transformer.

This script trains a production-grade Transformer model for
English-Hindi translation.

Usage:
    python scripts/train_nmt.py --config base
    python scripts/train_nmt.py --config small --epochs 10
    python scripts/train_nmt.py --resume models/translation/epoch_5.pt
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.nmt.config import NMTConfig, get_base_config, get_small_config, get_debug_config
from src.nmt.tokenizer import Tokenizer, train_nmt_tokenizer
from src.nmt.model.transformer import Transformer, create_model_from_config
from src.nmt.training.dataset import (
    TranslationDataset,
    TranslationDatasetStreaming,
    create_dataloader,
    create_streaming_dataloader
)
from src.nmt.training.trainer import Trainer
from src.nmt.training.utils import set_seed, get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Train NMT Transformer")
    
    # Config
    parser.add_argument("--config", type=str, default="base",
                       choices=["base", "small", "debug"],
                       help="Model configuration preset")
    
    # Data paths
    parser.add_argument("--train-data", type=str, 
                       default="data/raw/train-en-hi.json",
                       help="Path to training data JSON file")
    parser.add_argument("--val-data", type=str,
                       default="data/raw/validation-en-hi.json",
                       help="Path to validation data JSON file")
    
    # Tokenizer
    parser.add_argument("--tokenizer", type=str,
                       default="models/translation/nmt_spm.model",
                       help="Path to tokenizer model")
    parser.add_argument("--train-tokenizer", action="store_true",
                       help="Train a new tokenizer")
    parser.add_argument("--corpus", type=str,
                       default="data/raw/spm_corpus.txt",
                       help="Corpus for tokenizer training")
    
    # Training
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of epochs (overrides config)")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Batch size (overrides config)")
    parser.add_argument("--lr", type=float, default=None,
                       help="Learning rate (overrides config)")
    
    # Checkpointing
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--output-dir", type=str, 
                       default="models/translation",
                       help="Output directory for checkpoints")
    
    # Hardware
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cuda, cpu)")
    parser.add_argument("--no-amp", action="store_true",
                       help="Disable mixed precision training")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only run validation (requires --resume)")
    parser.add_argument("--streaming", action="store_true",
                       help="Use streaming dataset for large JSONL files (memory-efficient)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("NMT Transformer Training")
    print("=" * 60)
    
    # Set seed
    set_seed(args.seed)
    
    # Get device
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    print(f"Device: {device}")
    
    # Load configuration
    print(f"\nConfiguration: {args.config}")
    if args.config == "base":
        config = get_base_config()
    elif args.config == "small":
        config = get_small_config()
    else:
        config = get_debug_config()
    
    # Override config with CLI arguments
    if args.epochs is not None:
        config.training.max_epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.no_amp:
        config.training.use_amp = False
    
    config.model_dir = Path(args.output_dir)
    config.model_dir.mkdir(parents=True, exist_ok=True)
    
    # Train or load tokenizer
    print("\n--- Tokenizer ---")
    tokenizer_path = Path(args.tokenizer)
    
    if args.train_tokenizer or not tokenizer_path.exists():
        print("Training new tokenizer...")
        tokenizer = train_nmt_tokenizer(
            corpus_path=args.corpus,
            output_prefix=str(tokenizer_path).replace('.model', ''),
            vocab_size=config.tokenizer.vocab_size,
            language_tags=config.tokenizer.language_tags
        )
    else:
        print(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = Tokenizer(
            model_path=str(tokenizer_path),
            language_tags=config.tokenizer.language_tags
        )
    
    # Update config with tokenizer info
    config.model.vocab_size = tokenizer.vocab_size
    config.model.pad_id = tokenizer.pad_id
    config.model.unk_id = tokenizer.unk_id
    config.model.bos_id = tokenizer.bos_id
    config.model.eos_id = tokenizer.eos_id
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Load datasets
    print("\n--- Datasets ---")
    
    if args.streaming:
        print("Using STREAMING mode (memory-efficient for large files)")
        train_dataset = TranslationDatasetStreaming(
            data_path=args.train_data,
            tokenizer=tokenizer,
            max_length=config.model.max_seq_len,
            source_lang=f"<{config.source_lang}>",
            target_lang=f"<{config.target_lang}>"
        )
        print(f"Streaming dataset from {args.train_data}")
    else:
        train_dataset = TranslationDataset(
            data_path=args.train_data,
            tokenizer=tokenizer,
            max_length=config.model.max_seq_len,
            source_lang=f"<{config.source_lang}>",
            target_lang=f"<{config.target_lang}>"
        )
    
    val_dataset = None
    if Path(args.val_data).exists():
        val_dataset = TranslationDataset(
            data_path=args.val_data,
            tokenizer=tokenizer,
            max_length=config.model.max_seq_len,
            source_lang=f"<{config.source_lang}>",
            target_lang=f"<{config.target_lang}>"
        )
    
    # Create dataloaders
    if args.streaming:
        train_dataloader = create_streaming_dataloader(
            train_dataset,
            batch_size=config.training.batch_size,
            num_workers=0,  # Streaming works best with 0 workers
            pad_id=tokenizer.pad_id
        )
    else:
        train_dataloader = create_dataloader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=4,
            pad_id=tokenizer.pad_id
        )
    
    val_dataloader = None
    if val_dataset:
        val_dataloader = create_dataloader(
            val_dataset,
            batch_size=config.training.batch_size * 2,  # Larger for validation
            shuffle=False,
            num_workers=2,
            pad_id=tokenizer.pad_id
        )
    
    # Create model
    print("\n--- Model ---")
    model = create_model_from_config(config)
    print(f"Parameters: {model.count_parameters_readable()}")
    print(f"Architecture: {config.model.n_encoder_layers}L-{config.model.d_model}d-{config.model.n_heads}h")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        tokenizer=tokenizer,
        device=device
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Validate only mode
    if args.validate_only:
        if val_dataloader is None:
            print("Error: No validation data provided")
            return
        trainer.validate()
        return
    
    # Save config
    config.save(config.model_dir / "config.json")
    
    # Train
    print("\n--- Training ---")
    trainer.train()
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
