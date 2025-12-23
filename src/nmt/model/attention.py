"""
Multi-Head Attention Implementation.

Implements scaled dot-product attention and multi-head attention mechanism
as described in "Attention Is All You Need" (Vaswani et al., 2017).

Key features:
- Proper masking for padding and causal (autoregressive) attention
- Efficient batched attention computation
- Support for both self-attention and cross-attention
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism.
    
    Computes attention as:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    
    Multi-head attention runs multiple attention heads in parallel,
    each with different learned projections:
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_O
        where head_i = Attention(Q * W_Q_i, K * W_K_i, V * W_V_i)
    
    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        dropout: Attention dropout probability.
        bias: Whether to use bias in linear projections.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head
        self.scale = math.sqrt(self.d_k)
        
        # Linear projections for Q, K, V
        # Using single linear layers for efficiency
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        
        # Output projection
        self.w_o = nn.Linear(d_model, d_model, bias=bias)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform.
        
        Xavier initialization is preferred for attention weights
        as it maintains variance through the network.
        """
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute multi-head attention.
        
        Args:
            query: Query tensor of shape (batch, seq_len_q, d_model).
            key: Key tensor of shape (batch, seq_len_k, d_model).
            value: Value tensor of shape (batch, seq_len_k, d_model).
            mask: Attention mask of shape (batch, 1, seq_len_q, seq_len_k)
                  or (batch, 1, 1, seq_len_k) for key padding mask.
                  True/1 values indicate positions to ATTEND to.
                  False/0 values indicate positions to MASK.
            return_attention: Whether to return attention weights.
        
        Returns:
            Tuple of (output, attention_weights).
            - output: Shape (batch, seq_len_q, d_model).
            - attention_weights: Shape (batch, n_heads, seq_len_q, seq_len_k)
                                if return_attention=True, else None.
        """
        batch_size = query.size(0)
        
        # Linear projections: (batch, seq_len, d_model)
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)
        
        # Reshape to multi-head: (batch, n_heads, seq_len, d_k)
        q = q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        # Scores: (batch, n_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask for heads if needed
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (batch, 1, seq_len_q, seq_len_k)
            
            # Convert mask: True -> 0, False -> -inf
            # Our convention: 1 = attend, 0 = mask
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax over key dimension
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply dropout to attention weights
        attention_weights = self.dropout(attention_weights)
        
        # Compute weighted sum: (batch, n_heads, seq_len_q, d_k)
        context = torch.matmul(attention_weights, v)
        
        # Reshape back: (batch, seq_len_q, d_model)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Output projection
        output = self.w_o(context)
        
        if return_attention:
            return output, attention_weights
        return output, None


def create_padding_mask(
    seq: torch.Tensor,
    pad_idx: int = 0
) -> torch.Tensor:
    """Create padding mask for attention.
    
    The mask indicates which positions contain real tokens (1) vs padding (0).
    
    Args:
        seq: Token IDs of shape (batch, seq_len).
        pad_idx: Padding token index.
    
    Returns:
        Mask of shape (batch, 1, 1, seq_len) where:
        - 1 = real token (attend)
        - 0 = padding (mask)
    """
    # (batch, seq_len) -> (batch, 1, 1, seq_len)
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask.float()


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Create causal (autoregressive) mask for decoder self-attention.
    
    Prevents the decoder from attending to future positions.
    
    Args:
        seq_len: Sequence length.
        device: Device to create tensor on.
    
    Returns:
        Mask of shape (1, 1, seq_len, seq_len) where:
        - Lower triangle = 1 (attend)
        - Upper triangle = 0 (mask)
    
    Example for seq_len=4:
        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1]]
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)


def create_decoder_mask(
    tgt: torch.Tensor,
    pad_idx: int = 0
) -> torch.Tensor:
    """Create combined causal and padding mask for decoder.
    
    Combines:
    1. Causal mask (no attending to future)
    2. Padding mask (no attending to pad tokens)
    
    Args:
        tgt: Target token IDs of shape (batch, seq_len).
        pad_idx: Padding token index.
    
    Returns:
        Mask of shape (batch, 1, seq_len, seq_len).
    """
    batch_size, seq_len = tgt.size()
    device = tgt.device
    
    # Padding mask: (batch, 1, 1, seq_len)
    padding_mask = create_padding_mask(tgt, pad_idx)
    
    # Causal mask: (1, 1, seq_len, seq_len)
    causal_mask = create_causal_mask(seq_len, device)
    
    # Combine: broadcast and element-wise AND
    # Result: (batch, 1, seq_len, seq_len)
    combined_mask = padding_mask * causal_mask
    
    return combined_mask


def create_encoder_decoder_mask(
    src: torch.Tensor,
    tgt_len: int,
    pad_idx: int = 0
) -> torch.Tensor:
    """Create mask for encoder-decoder (cross) attention.
    
    Allows decoder to attend to all encoder positions except padding.
    
    Args:
        src: Source token IDs of shape (batch, src_len).
        tgt_len: Target sequence length.
        pad_idx: Padding token index.
    
    Returns:
        Mask of shape (batch, 1, tgt_len, src_len).
    """
    # Padding mask on source: (batch, 1, 1, src_len)
    src_mask = create_padding_mask(src, pad_idx)
    
    # Expand to (batch, 1, tgt_len, src_len)
    src_mask = src_mask.expand(-1, -1, tgt_len, -1)
    
    return src_mask
