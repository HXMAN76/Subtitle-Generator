"""
NMT Configuration Module.

Defines all hyperparameters and settings for the NMT system.
Uses dataclasses for type safety and easy serialization.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List
import json


@dataclass
class ModelConfig:
    """Transformer model architecture configuration.
    
    Base configuration (6L-512d) is tuned for ~6M sentence pairs.
    For smaller datasets (<1M), consider 4L-256d variant.
    
    Reference: Vaswani et al., 2017 "Attention Is All You Need"
    """
    
    # Vocabulary
    vocab_size: int = 32000
    
    # Model dimensions
    d_model: int = 512          # Hidden dimension (must be divisible by n_heads)
    n_heads: int = 8            # Number of attention heads
    d_ff: int = 2048            # Feed-forward dimension (typically 4 * d_model)
    
    # Architecture depth
    n_encoder_layers: int = 6
    n_decoder_layers: int = 6
    
    # Regularization
    dropout: float = 0.1       # 0.1 standard; 0.3 for <1M data
    attention_dropout: float = 0.1
    
    # Sequence length
    max_seq_len: int = 256
    
    # Special token IDs (set by tokenizer)
    pad_id: int = 0
    unk_id: int = 1
    bos_id: int = 2
    eos_id: int = 3
    
    # Weight tying: share embeddings between encoder, decoder, and output layer
    # Reduces parameters by ~30% with minimal quality loss (Press & Wolf, 2017)
    tie_embeddings: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        assert self.d_ff >= self.d_model, \
            f"d_ff ({self.d_ff}) should be >= d_model ({self.d_model})"


@dataclass
class TrainingConfig:
    """Training hyperparameters.
    
    Defaults optimized for Transformer training on 6M+ sentence pairs.
    """
    
    # Batch size (adjust based on GPU memory)
    # RTX 4060: ~32-48 effective batch size with gradient accumulation
    # RTX 4000 Ada: ~64-96 effective batch size
    batch_size: int = 32
    gradient_accumulation_steps: int = 2  # Effective batch = 64
    
    # Learning rate with warmup
    # Peak LR = d_model^(-0.5) * warmup^(-0.5) ≈ 7e-4 for d_model=512, warmup=4000
    learning_rate: float = 1e-4
    warmup_steps: int = 4000
    
    # Optimizer (AdamW with Transformer-specific betas)
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98     # Lower than default 0.999 for Transformers
    adam_epsilon: float = 1e-9
    weight_decay: float = 0.01
    
    # Gradient clipping (prevents explosion in early training)
    max_grad_norm: float = 1.0
    
    # Training duration
    max_epochs: int = 30
    max_steps: Optional[int] = None  # If set, overrides max_epochs
    
    # Regularization
    label_smoothing: float = 0.1  # Improves BLEU by 0.5-1.0 (Szegedy et al.)
    
    # Checkpointing
    save_every_n_steps: int = 5000
    validate_every_n_steps: int = 1000
    
    # Early stopping
    patience: int = 5            # Stop if no improvement for N epochs
    min_delta: float = 0.001     # Minimum improvement threshold
    
    # Mixed precision (AMP)
    use_amp: bool = True         # ~2x speedup, ~50% memory reduction
    
    # Reproducibility
    seed: int = 42


@dataclass
class TokenizerConfig:
    """SentencePiece tokenizer configuration."""
    
    vocab_size: int = 32000
    model_type: str = "bpe"      # "bpe" or "unigram"
    character_coverage: float = 0.9995  # Higher for Devanagari script
    
    # Special tokens
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    
    # Language tags for multilingual support
    language_tags: List[str] = field(default_factory=lambda: ["<en>", "<hi>"])
    
    # Paths
    model_prefix: str = "models/translation/nmt_spm"
    corpus_path: str = "data/raw/spm_corpus.txt"


@dataclass 
class NMTConfig:
    """Complete NMT system configuration."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    
    # Paths
    data_dir: Path = Path("data")
    model_dir: Path = Path("models/translation")
    output_dir: Path = Path("output")
    
    # Language pair
    source_lang: str = "en"
    target_lang: str = "hi"
    
    def save(self, path: Path):
        """Save configuration to JSON file."""
        config_dict = {
            "model": asdict(self.model),
            "training": asdict(self.training),
            "tokenizer": asdict(self.tokenizer),
            "data_dir": str(self.data_dir),
            "model_dir": str(self.model_dir),
            "output_dir": str(self.output_dir),
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: Path) -> "NMTConfig":
        """Load configuration from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        
        return cls(
            model=ModelConfig(**config_dict["model"]),
            training=TrainingConfig(**config_dict["training"]),
            tokenizer=TokenizerConfig(**config_dict["tokenizer"]),
            data_dir=Path(config_dict["data_dir"]),
            model_dir=Path(config_dict["model_dir"]),
            output_dir=Path(config_dict["output_dir"]),
            source_lang=config_dict["source_lang"],
            target_lang=config_dict["target_lang"],
        )


# Default configurations for different scales
def get_base_config() -> NMTConfig:
    """Base configuration for 6M+ sentence pairs (RTX 4000 Ada)."""
    return NMTConfig()


def get_small_config() -> NMTConfig:
    """Small configuration for <1M sentence pairs or quick iteration."""
    config = NMTConfig()
    config.model.d_model = 256
    config.model.n_heads = 4
    config.model.d_ff = 1024
    config.model.n_encoder_layers = 4
    config.model.n_decoder_layers = 4
    config.model.dropout = 0.2
    config.training.batch_size = 64
    config.training.warmup_steps = 2000
    return config


def get_debug_config() -> NMTConfig:
    """Minimal configuration for debugging and testing."""
    config = NMTConfig()
    config.model.d_model = 64
    config.model.n_heads = 2
    config.model.d_ff = 256
    config.model.n_encoder_layers = 2
    config.model.n_decoder_layers = 2
    config.model.max_seq_len = 64
    config.training.batch_size = 8
    config.training.warmup_steps = 100
    config.training.max_epochs = 2
    config.training.use_amp = False
    return config
