"""
Transformer Decoder Implementation.

Implements the decoder stack from "Attention Is All You Need":
- N identical layers
- Each layer: Masked Self-Attention -> Cross-Attention -> FFN
- All with Add & Norm residual connections

Uses Pre-LN (Layer Norm before sublayer) for training stability.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    """Single Transformer Decoder Layer.
    
    Architecture (Pre-LN):
        x -> LayerNorm -> MaskedSelfAttention -> Dropout -> Add (residual)
          -> LayerNorm -> CrossAttention(x, encoder_out) -> Dropout -> Add
          -> LayerNorm -> FFN -> Dropout -> Add (residual)
    
    The decoder has three sublayers:
        1. Masked self-attention (causal, no peeking at future tokens)
        2. Encoder-decoder attention (cross-attention)
        3. Position-wise feed-forward network
    
    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        d_ff: Feed-forward hidden dimension.
        dropout: Dropout probability.
        attention_dropout: Attention-specific dropout.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1
    ):
        super().__init__()
        
        # Masked self-attention sublayer
        self.self_attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=attention_dropout
        )
        
        # Encoder-decoder (cross) attention sublayer
        self.cross_attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=attention_dropout
        )
        
        # Feed-forward sublayer
        self.feed_forward = PositionwiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )
        
        # Layer normalization (Pre-LN)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm3 = nn.LayerNorm(d_model, eps=1e-6)
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through decoder layer.
        
        Args:
            x: Decoder input of shape (batch, tgt_len, d_model).
            encoder_output: Encoder output of shape (batch, src_len, d_model).
            self_attn_mask: Causal + padding mask for self-attention.
                           Shape: (batch, 1, tgt_len, tgt_len).
            cross_attn_mask: Padding mask for cross-attention.
                            Shape: (batch, 1, tgt_len, src_len).
        
        Returns:
            Output tensor of shape (batch, tgt_len, d_model).
        """
        # Masked self-attention sublayer
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attention(x, x, x, mask=self_attn_mask)
        x = self.dropout(x)
        x = residual + x
        
        # Cross-attention sublayer
        residual = x
        x = self.norm2(x)
        x, _ = self.cross_attention(x, encoder_output, encoder_output, mask=cross_attn_mask)
        x = self.dropout(x)
        x = residual + x
        
        # Feed-forward sublayer
        residual = x
        x = self.norm3(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = residual + x
        
        return x
    
    def forward_with_cache(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
        cache: Optional[dict] = None
    ) -> Tuple[torch.Tensor, dict]:
        """Forward pass with KV cache for efficient autoregressive decoding.
        
        During inference, we cache key/value tensors from previous steps
        to avoid recomputing them.
        
        Args:
            x: Current step input of shape (batch, 1, d_model).
            encoder_output: Encoder output.
            self_attn_mask: Self-attention mask.
            cross_attn_mask: Cross-attention mask.
            cache: Dictionary with cached K, V tensors.
        
        Returns:
            Tuple of (output, updated_cache).
        """
        if cache is None:
            cache = {}
        
        # Masked self-attention with cache
        residual = x
        x = self.norm1(x)
        
        # Use cached K, V for self-attention
        if 'self_k' in cache:
            # Append current K, V to cache
            k = torch.cat([cache['self_k'], x], dim=1)
            v = torch.cat([cache['self_v'], x], dim=1)
        else:
            k = x
            v = x
        
        cache['self_k'] = k
        cache['self_v'] = v
        
        x, _ = self.self_attention(x, k, v, mask=self_attn_mask)
        x = self.dropout(x)
        x = residual + x
        
        # Cross-attention (no caching needed, encoder output is static)
        residual = x
        x = self.norm2(x)
        x, _ = self.cross_attention(x, encoder_output, encoder_output, mask=cross_attn_mask)
        x = self.dropout(x)
        x = residual + x
        
        # Feed-forward
        residual = x
        x = self.norm3(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = residual + x
        
        return x, cache


class TransformerDecoder(nn.Module):
    """Transformer Decoder Stack.
    
    Stacks N identical DecoderLayers with a final layer normalization.
    
    Args:
        n_layers: Number of decoder layers.
        d_model: Model dimension.
        n_heads: Number of attention heads.
        d_ff: Feed-forward hidden dimension.
        dropout: Dropout probability.
        attention_dropout: Attention-specific dropout.
    """
    
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1
    ):
        super().__init__()
        
        self.n_layers = n_layers
        self.d_model = d_model
        
        # Stack of decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                attention_dropout=attention_dropout
            )
            for _ in range(n_layers)
        ])
        
        # Final layer normalization
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through decoder stack.
        
        Args:
            x: Decoder input embeddings of shape (batch, tgt_len, d_model).
            encoder_output: Encoder output of shape (batch, src_len, d_model).
            self_attn_mask: Causal + padding mask for self-attention.
            cross_attn_mask: Padding mask for cross-attention.
        
        Returns:
            Decoder output of shape (batch, tgt_len, d_model).
        """
        # Pass through each decoder layer
        for layer in self.layers:
            x = layer(x, encoder_output, self_attn_mask, cross_attn_mask)
        
        # Final layer normalization
        x = self.norm(x)
        
        return x
    
    def forward_with_cache(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
        caches: Optional[list] = None
    ) -> Tuple[torch.Tensor, list]:
        """Forward pass with KV caching for autoregressive decoding.
        
        Args:
            x: Current step input of shape (batch, 1, d_model).
            encoder_output: Encoder output.
            self_attn_mask: Self-attention mask.
            cross_attn_mask: Cross-attention mask.
            caches: List of cache dictionaries, one per layer.
        
        Returns:
            Tuple of (output, updated_caches).
        """
        if caches is None:
            caches = [None] * self.n_layers
        
        new_caches = []
        
        for layer, cache in zip(self.layers, caches):
            x, new_cache = layer.forward_with_cache(
                x, encoder_output, self_attn_mask, cross_attn_mask, cache
            )
            new_caches.append(new_cache)
        
        x = self.norm(x)
        
        return x, new_caches
