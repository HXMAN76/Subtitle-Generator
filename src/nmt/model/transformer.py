"""
Complete Transformer Model for Neural Machine Translation.

Combines encoder, decoder, embeddings, and output projection into
a complete sequence-to-sequence Transformer model.

Features:
- Weight tying between embeddings and output projection
- Proper initialization
- Inference-ready with integrated masking
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

from .embeddings import TransformerEmbeddings
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder
from .attention import (
    create_padding_mask,
    create_decoder_mask,
    create_encoder_decoder_mask
)


class Transformer(nn.Module):
    """Complete Transformer for Neural Machine Translation.
    
    This is the main model class that combines all components:
    - Source and target embeddings (optionally shared)
    - Transformer encoder
    - Transformer decoder
    - Output projection (optionally tied with embeddings)
    
    Architecture follows "Attention Is All You Need" with improvements:
    - Pre-LN for training stability
    - Optional weight tying for parameter efficiency
    
    Args:
        vocab_size: Vocabulary size (shared for source and target).
        d_model: Model dimension.
        n_heads: Number of attention heads.
        n_encoder_layers: Number of encoder layers.
        n_decoder_layers: Number of decoder layers.
        d_ff: Feed-forward hidden dimension.
        max_seq_len: Maximum sequence length.
        dropout: Dropout probability.
        attention_dropout: Attention-specific dropout.
        pad_idx: Padding token index.
        tie_embeddings: Whether to tie encoder/decoder/output embeddings.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        pad_idx: int = 0,
        tie_embeddings: bool = True
    ):
        super().__init__()
        
        # Store config
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.pad_idx = pad_idx
        self.tie_embeddings = tie_embeddings
        
        # Embeddings
        self.src_embedding = TransformerEmbeddings(
            vocab_size=vocab_size,
            d_model=d_model,
            max_len=max_seq_len,
            dropout=dropout,
            pad_idx=pad_idx
        )
        
        if tie_embeddings:
            # Share embeddings between encoder and decoder
            self.tgt_embedding = self.src_embedding
        else:
            self.tgt_embedding = TransformerEmbeddings(
                vocab_size=vocab_size,
                d_model=d_model,
                max_len=max_seq_len,
                dropout=dropout,
                pad_idx=pad_idx
            )
        
        # Encoder
        self.encoder = TransformerEncoder(
            n_layers=n_encoder_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            attention_dropout=attention_dropout
        )
        
        # Decoder
        self.decoder = TransformerDecoder(
            n_layers=n_decoder_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            attention_dropout=attention_dropout
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        
        if tie_embeddings:
            # Tie output projection to embedding weights
            # This reduces parameters and often improves performance
            # (Press & Wolf, 2017 - "Using the Output Embedding to Improve Language Models")
            self.output_projection.weight = self.src_embedding.weight
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform.
        
        Output projection is initialized separately if not tied.
        """
        if not self.tie_embeddings:
            nn.init.xavier_uniform_(self.output_projection.weight)
    
    def encode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode source sequence.
        
        Args:
            src: Source token IDs of shape (batch, src_len).
            src_mask: Source padding mask.
        
        Returns:
            Encoder output of shape (batch, src_len, d_model).
        """
        # Create source mask if not provided
        if src_mask is None:
            src_mask = create_padding_mask(src, self.pad_idx)
        
        # Embed and encode
        src_emb = self.src_embedding(src)
        encoder_output = self.encoder(src_emb, mask=src_mask)
        
        return encoder_output
    
    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        src: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Decode target sequence.
        
        Args:
            tgt: Target token IDs of shape (batch, tgt_len).
            encoder_output: Encoder output of shape (batch, src_len, d_model).
            src: Source token IDs (for cross-attention mask).
            tgt_mask: Target causal + padding mask.
            cross_mask: Cross-attention mask.
        
        Returns:
            Decoder output of shape (batch, tgt_len, d_model).
        """
        tgt_len = tgt.size(1)
        
        # Create masks if not provided
        if tgt_mask is None:
            tgt_mask = create_decoder_mask(tgt, self.pad_idx)
        
        if cross_mask is None:
            cross_mask = create_encoder_decoder_mask(src, tgt_len, self.pad_idx)
        
        # Embed and decode
        tgt_emb = self.tgt_embedding(tgt)
        decoder_output = self.decoder(
            tgt_emb, encoder_output, tgt_mask, cross_mask
        )
        
        return decoder_output
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Full forward pass for training.
        
        Args:
            src: Source token IDs of shape (batch, src_len).
            tgt: Target token IDs of shape (batch, tgt_len).
            src_mask: Source padding mask.
            tgt_mask: Target causal + padding mask.
            cross_mask: Cross-attention mask.
        
        Returns:
            Logits of shape (batch, tgt_len, vocab_size).
        """
        # Encode source
        encoder_output = self.encode(src, src_mask)
        
        # Decode target
        decoder_output = self.decode(
            tgt, encoder_output, src, tgt_mask, cross_mask
        )
        
        # Project to vocabulary
        logits = self.output_projection(decoder_output)
        
        return logits
    
    def generate_step(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        src: torch.Tensor,
        caches: Optional[list] = None
    ) -> Tuple[torch.Tensor, list]:
        """Single generation step with KV caching.
        
        Used during inference for efficient autoregressive decoding.
        
        Args:
            tgt: Current target token(s) of shape (batch, 1) or (batch, seq_len).
            encoder_output: Encoder output.
            src: Source tokens (for mask creation).
            caches: List of KV caches from previous steps.
        
        Returns:
            Tuple of (logits, updated_caches).
        """
        tgt_len = tgt.size(1)
        
        # Create masks
        tgt_mask = create_decoder_mask(tgt, self.pad_idx)
        cross_mask = create_encoder_decoder_mask(src, tgt_len, self.pad_idx)
        
        # Embed target
        tgt_emb = self.tgt_embedding(tgt)
        
        # Decode with caching
        decoder_output, caches = self.decoder.forward_with_cache(
            tgt_emb, encoder_output, tgt_mask, cross_mask, caches
        )
        
        # Project to vocabulary (only last position)
        logits = self.output_projection(decoder_output[:, -1:, :])
        
        return logits.squeeze(1), caches
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration as dictionary."""
        return {
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_encoder_layers': self.n_encoder_layers,
            'n_decoder_layers': self.n_decoder_layers,
            'd_ff': self.d_ff,
            'max_seq_len': self.max_seq_len,
            'pad_idx': self.pad_idx,
            'tie_embeddings': self.tie_embeddings,
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'Transformer':
        """Create model from configuration dictionary."""
        return cls(**config)
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def count_parameters_readable(self) -> str:
        """Get human-readable parameter count."""
        n = self.count_parameters()
        if n >= 1e9:
            return f"{n / 1e9:.2f}B"
        elif n >= 1e6:
            return f"{n / 1e6:.2f}M"
        elif n >= 1e3:
            return f"{n / 1e3:.2f}K"
        return str(n)


def create_model_from_config(config) -> Transformer:
    """Create Transformer model from NMTConfig.
    
    Args:
        config: NMTConfig object.
    
    Returns:
        Initialized Transformer model.
    """
    model_cfg = config.model
    
    model = Transformer(
        vocab_size=model_cfg.vocab_size,
        d_model=model_cfg.d_model,
        n_heads=model_cfg.n_heads,
        n_encoder_layers=model_cfg.n_encoder_layers,
        n_decoder_layers=model_cfg.n_decoder_layers,
        d_ff=model_cfg.d_ff,
        max_seq_len=model_cfg.max_seq_len,
        dropout=model_cfg.dropout,
        attention_dropout=model_cfg.attention_dropout,
        pad_idx=model_cfg.pad_id,
        tie_embeddings=model_cfg.tie_embeddings
    )
    
    return model
