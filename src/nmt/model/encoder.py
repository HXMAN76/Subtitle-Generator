"""
Transformer Encoder Implementation.

Implements the encoder stack from "Attention Is All You Need":
- N identical layers
- Each layer: Self-Attention -> Add & Norm -> FFN -> Add & Norm

Uses Pre-LN (Layer Norm before sublayer) which is more stable for training
compared to Post-LN from the original paper.

Reference:
- "On Layer Normalization in the Transformer Architecture" (Xiong et al., 2020)
"""

import torch
import torch.nn as nn
from typing import Optional

from .attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    """Single Transformer Encoder Layer.
    
    Architecture (Pre-LN):
        x -> LayerNorm -> SelfAttention -> Dropout -> Add (residual)
          -> LayerNorm -> FFN -> Dropout -> Add (residual)
    
    Pre-LN is preferred over Post-LN for:
        1. Better gradient flow
        2. More stable training
        3. Allows higher learning rates
    
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
        
        # Self-attention sublayer
        self.self_attention = MultiHeadAttention(
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
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through encoder layer.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            mask: Attention mask of shape (batch, 1, 1, seq_len).
        
        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        # Self-attention sublayer with Pre-LN and residual
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attention(x, x, x, mask=mask)
        x = self.dropout(x)
        x = residual + x
        
        # Feed-forward sublayer with Pre-LN and residual
        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = residual + x
        
        return x


class TransformerEncoder(nn.Module):
    """Transformer Encoder Stack.
    
    Stacks N identical EncoderLayers with a final layer normalization.
    
    Args:
        n_layers: Number of encoder layers.
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
        
        # Stack of encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(
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
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through encoder stack.
        
        Args:
            x: Input embeddings of shape (batch, seq_len, d_model).
            mask: Attention mask of shape (batch, 1, 1, seq_len).
        
        Returns:
            Encoder output of shape (batch, seq_len, d_model).
        """
        # Pass through each encoder layer
        for layer in self.layers:
            x = layer(x, mask=mask)
        
        # Final layer normalization
        x = self.norm(x)
        
        return x
    
    def get_attention_weights(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> list:
        """Get attention weights from all layers (for visualization).
        
        Args:
            x: Input embeddings.
            mask: Attention mask.
        
        Returns:
            List of attention weight tensors, one per layer.
        """
        attention_weights = []
        
        for layer in self.layers:
            residual = x
            x = layer.norm1(x)
            x, attn = layer.self_attention(x, x, x, mask=mask, return_attention=True)
            attention_weights.append(attn)
            x = layer.dropout(x)
            x = residual + x
            
            residual = x
            x = layer.norm2(x)
            x = layer.feed_forward(x)
            x = layer.dropout(x)
            x = residual + x
        
        return attention_weights
