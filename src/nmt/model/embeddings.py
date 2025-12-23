"""
Positional Encoding and Token Embeddings for Transformer.

Implements:
- Sinusoidal positional encoding (Vaswani et al., 2017)
- Scaled token embeddings
- Combined embedding layer with dropout
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding.
    
    Adds positional information to token embeddings using sin/cos functions.
    This is the original approach from "Attention Is All You Need".
    
    Formula:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Why sinusoidal over learned:
        1. Generalizes to unseen sequence lengths
        2. Relative position can be represented as linear function
        3. No additional parameters to learn
        4. Works well empirically for NMT tasks
    
    Args:
        d_model: Model dimension.
        max_len: Maximum sequence length.
        dropout: Dropout probability.
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute div_term: 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings.
        
        Args:
            x: Token embeddings of shape (batch, seq_len, d_model).
        
        Returns:
            Embeddings with positional encoding added.
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class TransformerEmbeddings(nn.Module):
    """Combined Token and Positional Embeddings.
    
    This module handles:
    1. Token embedding lookup (with scaling by sqrt(d_model))
    2. Positional encoding addition
    3. Dropout
    
    Embedding scaling (sqrt(d_model)) is crucial:
        - Prevents embedding values from being too small relative to positional encoding
        - Empirically shown to improve training stability
        - From original Transformer paper
    
    Args:
        vocab_size: Size of vocabulary.
        d_model: Model dimension.
        max_len: Maximum sequence length.
        dropout: Dropout probability.
        pad_idx: Padding token index (embeddings will be zero for this index).
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
        pad_idx: int = 0
    ):
        super().__init__()
        
        self.d_model = d_model
        self.scale = math.sqrt(d_model)
        
        # Token embeddings
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=pad_idx
        )
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=max_len,
            dropout=dropout
        )
        
        # Initialize embeddings
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embedding weights.
        
        Using normal initialization with std=0.02 following GPT/BERT convention.
        """
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        # Zero out padding embedding
        if self.token_embedding.padding_idx is not None:
            self.token_embedding.weight.data[self.token_embedding.padding_idx].zero_()
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed tokens with positional encoding.
        
        Args:
            tokens: Token IDs of shape (batch, seq_len).
        
        Returns:
            Embeddings of shape (batch, seq_len, d_model).
        """
        # Token embeddings scaled by sqrt(d_model)
        embeddings = self.token_embedding(tokens) * self.scale
        
        # Add positional encoding (includes dropout)
        embeddings = self.positional_encoding(embeddings)
        
        return embeddings
    
    @property
    def weight(self) -> torch.Tensor:
        """Return embedding weight matrix (for weight tying)."""
        return self.token_embedding.weight


class RotaryPositionalEncoding(nn.Module):
    """Rotary Position Embedding (RoPE).
    
    Alternative to sinusoidal encoding from Su et al., "RoFormer".
    Benefits:
        - Better relative position modeling
        - More flexible for varying sequence lengths
        - Widely used in modern LLMs (LLaMA, etc.)
    
    Note: Included for future use but not default choice for NMT.
    Sinusoidal is preferred for translation due to proven effectiveness.
    
    Args:
        d_model: Model dimension.
        max_len: Maximum sequence length.
        base: Base for frequency computation.
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        base: int = 10000
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        self.base = base
        
        # Precompute rotation matrices
        self._precompute_freqs(max_len)
    
    def _precompute_freqs(self, max_len: int):
        """Precompute frequency tensors for rotation."""
        dim = self.d_model // 2
        
        # Frequencies: 1 / (base^(2i/d) for i in [0, d/2))
        freqs = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        
        # Position indices
        t = torch.arange(max_len, dtype=torch.float)
        
        # Outer product: (max_len, dim/2)
        freqs = torch.outer(t, freqs)
        
        # Complex representation for rotation
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        
        self.register_buffer('freqs_cis', freqs_cis)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotary position encoding.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
        
        Returns:
            Position-encoded tensor.
        """
        seq_len = x.size(1)
        
        # Reshape to complex pairs
        x_complex = torch.view_as_complex(
            x.float().reshape(*x.shape[:-1], -1, 2)
        )
        
        # Apply rotation
        freqs = self.freqs_cis[:seq_len].unsqueeze(0)
        x_rotated = x_complex * freqs
        
        # Convert back to real
        x_out = torch.view_as_real(x_rotated).flatten(-2)
        
        return x_out.type_as(x)
