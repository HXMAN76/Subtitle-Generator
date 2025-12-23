"""
Position-wise Feed-Forward Network.

Implements the FFN sublayer from "Attention Is All You Need":
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2

This is applied to each position independently and identically.
The expansion ratio (d_ff / d_model) is typically 4.
"""

import torch
import torch.nn as nn
from typing import Optional


class PositionwiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Network.
    
    A two-layer feed-forward network with ReLU/GELU activation.
    Applied independently to each position in the sequence.
    
    Architecture:
        Linear(d_model -> d_ff) -> Activation -> Dropout -> Linear(d_ff -> d_model)
    
    The standard configuration uses d_ff = 4 * d_model.
    
    Args:
        d_model: Model dimension (input and output).
        d_ff: Feed-forward hidden dimension.
        dropout: Dropout probability.
        activation: Activation function ("relu" or "gelu").
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        
        # First linear layer (expansion)
        self.w_1 = nn.Linear(d_model, d_ff)
        
        # Second linear layer (projection)
        self.w_2 = nn.Linear(d_ff, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        nn.init.xavier_uniform_(self.w_1.weight)
        nn.init.xavier_uniform_(self.w_2.weight)
        nn.init.zeros_(self.w_1.bias)
        nn.init.zeros_(self.w_2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply position-wise feed-forward network.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
        
        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        # Expand -> Activation -> Dropout -> Project
        x = self.w_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.w_2(x)
        
        return x


class GatedFeedForward(nn.Module):
    """Gated Linear Unit (GLU) variant of FFN.
    
    From "GLU Variants Improve Transformer" (Shazeer, 2020).
    Uses a gating mechanism for better gradient flow.
    
    Architecture:
        GLU(x) = (xW_1) * σ(xW_gate)
        FFN_GLU(x) = (GLU(x))W_2
    
    Note: Included for potential future improvement but not default.
    Standard FFN with ReLU is proven for NMT tasks.
    
    Args:
        d_model: Model dimension.
        d_ff: Feed-forward hidden dimension.
        dropout: Dropout probability.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Gate and projection (combined for efficiency)
        self.w_gate = nn.Linear(d_model, d_ff)
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in [self.w_gate, self.w_1, self.w_2]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gated feed-forward network.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
        
        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        # Gated Linear Unit
        gate = torch.sigmoid(self.w_gate(x))
        hidden = self.w_1(x) * gate
        
        # Project back
        hidden = self.dropout(hidden)
        output = self.w_2(hidden)
        
        return output
