"""
Greedy Decoding for NMT.

Simple, fast decoding strategy that always picks the most likely token.
Used for:
- Fast inference when quality is less critical
- Debugging and testing
- Baseline comparisons
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any


@torch.no_grad()
def greedy_decode(
    model,
    src: torch.Tensor,
    bos_id: int,
    eos_id: int,
    pad_id: int = 0,
    max_length: int = 256,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """Greedy decoding.
    
    At each step, select the token with highest probability.
    Fast but suboptimal compared to beam search.
    
    Args:
        model: Transformer model.
        src: Source token IDs of shape (batch, src_len).
        bos_id: Beginning of sequence token ID.
        eos_id: End of sequence token ID.
        pad_id: Padding token ID.
        max_length: Maximum output length.
        device: Device for computation.
    
    Returns:
        Generated token IDs of shape (batch, out_len).
    """
    if device is None:
        device = src.device
    
    batch_size = src.size(0)
    model.eval()
    
    # Encode source
    encoder_output = model.encode(src)
    
    # Initialize decoder input with BOS token
    decoder_input = torch.full(
        (batch_size, 1),
        bos_id,
        dtype=torch.long,
        device=device
    )
    
    # Track which sequences are finished
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    # Generate tokens autoregressively
    for _ in range(max_length - 1):
        # Decode current sequence
        decoder_output = model.decode(
            decoder_input, encoder_output, src
        )
        
        # Get logits for last position
        logits = model.output_projection(decoder_output[:, -1, :])
        
        # Greedy selection: pick token with highest probability
        next_token = logits.argmax(dim=-1, keepdim=True)  # (batch, 1)
        
        # Append to decoder input
        decoder_input = torch.cat([decoder_input, next_token], dim=1)
        
        # Check for EOS
        finished = finished | (next_token.squeeze(-1) == eos_id)
        
        # Stop if all sequences finished
        if finished.all():
            break
    
    return decoder_input


@torch.no_grad()
def greedy_decode_with_cache(
    model,
    src: torch.Tensor,
    bos_id: int,
    eos_id: int,
    max_length: int = 256,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """Greedy decoding with KV cache for faster inference.
    
    Uses cached key/value tensors to avoid redundant computation.
    Significantly faster for longer sequences.
    
    Args:
        model: Transformer model.
        src: Source token IDs of shape (batch, src_len).
        bos_id: Beginning of sequence token ID.
        eos_id: End of sequence token ID.
        max_length: Maximum output length.
        device: Device for computation.
    
    Returns:
        Generated token IDs of shape (batch, out_len).
    """
    if device is None:
        device = src.device
    
    batch_size = src.size(0)
    model.eval()
    
    # Encode source (done once)
    encoder_output = model.encode(src)
    
    # Initialize
    current_token = torch.full(
        (batch_size, 1),
        bos_id,
        dtype=torch.long,
        device=device
    )
    
    generated = [current_token]
    caches = None
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    for _ in range(max_length - 1):
        # Decode with caching
        logits, caches = model.generate_step(
            current_token, encoder_output, src, caches
        )
        
        # Greedy selection
        next_token = logits.argmax(dim=-1, keepdim=True)
        generated.append(next_token)
        
        # Check for EOS
        finished = finished | (next_token.squeeze(-1) == eos_id)
        if finished.all():
            break
        
        current_token = next_token
    
    # Concatenate all generated tokens
    output = torch.cat(generated, dim=1)
    
    return output
