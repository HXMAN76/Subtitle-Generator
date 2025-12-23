"""
Neural Machine Translation (NMT) Subsystem.

Production-grade Transformer-based translation system for multilingual support.

Modules:
    - model: Transformer encoder-decoder architecture
    - training: Training pipeline with warmup scheduling
    - inference: Beam search and greedy decoding
    - evaluation: BLEU, METEOR, COMET metrics
"""

from .config import NMTConfig, ModelConfig
from .tokenizer import Tokenizer

__version__ = "1.0.0"
__all__ = ["NMTConfig", "ModelConfig", "Tokenizer"]
