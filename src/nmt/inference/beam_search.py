"""
Beam Search Decoding for NMT.

Production-quality beam search with:
- Length normalization (Wu et al., 2016)
- Early stopping
- Batch beam search for efficiency

Beam search typically improves BLEU by 1-2 points over greedy decoding.
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class BeamHypothesis:
    """A single beam hypothesis."""
    tokens: List[int]
    score: float
    
    def __len__(self):
        return len(self.tokens)


class BeamSearchDecoder:
    """Beam search decoder with length normalization.
    
    Implements beam search following Google's NMT paper:
    "Google's Neural Machine Translation System" (Wu et al., 2016)
    
    Key features:
    - Length normalization to prevent short output bias
    - n-best list output
    - Early termination optimization
    
    Args:
        model: Transformer model.
        beam_size: Number of beams to maintain.
        max_length: Maximum output length.
        length_penalty: Length normalization factor (α).
        bos_id: Beginning of sequence token ID.
        eos_id: End of sequence token ID.
        pad_id: Padding token ID.
    """
    
    def __init__(
        self,
        model,
        beam_size: int = 4,
        max_length: int = 256,
        length_penalty: float = 0.6,
        bos_id: int = 2,
        eos_id: int = 3,
        pad_id: int = 0
    ):
        self.model = model
        self.beam_size = beam_size
        self.max_length = max_length
        self.length_penalty = length_penalty
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
    
    def length_norm(self, length: int) -> float:
        """Compute length normalization factor.
        
        Formula from Wu et al., 2016:
            lp(Y) = (5 + |Y|)^α / (5 + 1)^α
        
        Args:
            length: Sequence length.
        
        Returns:
            Normalization factor.
        """
        return ((5 + length) ** self.length_penalty) / ((5 + 1) ** self.length_penalty)
    
    @torch.no_grad()
    def decode(
        self,
        src: torch.Tensor,
        n_best: int = 1
    ) -> List[List[BeamHypothesis]]:
        """Beam search decoding.
        
        Args:
            src: Source token IDs of shape (batch, src_len).
            n_best: Number of hypotheses to return per input.
        
        Returns:
            List of n_best hypotheses for each batch item.
        """
        self.model.eval()
        device = src.device
        batch_size = src.size(0)
        
        # Encode source
        encoder_output = self.model.encode(src)
        
        # Expand for beam search: (batch * beam, src_len, d_model)
        encoder_output = encoder_output.unsqueeze(1).repeat(
            1, self.beam_size, 1, 1
        ).view(batch_size * self.beam_size, -1, encoder_output.size(-1))
        
        src_expanded = src.unsqueeze(1).repeat(
            1, self.beam_size, 1
        ).view(batch_size * self.beam_size, -1)
        
        # Initialize beams
        # Shape: (batch * beam, seq_len)
        alive_seq = torch.full(
            (batch_size * self.beam_size, 1),
            self.bos_id,
            dtype=torch.long,
            device=device
        )
        
        # Log probabilities: (batch * beam,)
        alive_log_probs = torch.zeros(batch_size * self.beam_size, device=device)
        # First beam is active, others are dead (-inf)
        alive_log_probs[1::self.beam_size] = float('-inf')
        
        # Finished hypotheses for each batch item
        finished: List[List[BeamHypothesis]] = [[] for _ in range(batch_size)]
        
        # Run beam search
        for step in range(self.max_length - 1):
            # Decode current sequences
            decoder_output = self.model.decode(
                alive_seq, encoder_output, src_expanded
            )
            
            # Get logits for last position: (batch * beam, vocab_size)
            logits = self.model.output_projection(decoder_output[:, -1, :])
            log_probs = F.log_softmax(logits, dim=-1)
            
            vocab_size = log_probs.size(-1)
            
            # Compute scores for all possible extensions
            # (batch * beam, vocab) + (batch * beam, 1) -> (batch * beam, vocab)
            scores = log_probs + alive_log_probs.unsqueeze(-1)
            
            # Reshape to (batch, beam * vocab)
            scores = scores.view(batch_size, -1)
            
            # Select top 2 * beam candidates (allows for EOS candidates)
            topk_scores, topk_ids = scores.topk(2 * self.beam_size, dim=-1)
            
            # Convert flat indices to (beam_idx, token_idx)
            topk_beam_idx = topk_ids // vocab_size
            topk_token_idx = topk_ids % vocab_size
            
            # Build new alive sequences
            new_alive_seq = []
            new_alive_scores = []
            
            for batch_idx in range(batch_size):
                batch_offset = batch_idx * self.beam_size
                
                candidates = []
                for k in range(2 * self.beam_size):
                    beam_idx = topk_beam_idx[batch_idx, k].item()
                    token_idx = topk_token_idx[batch_idx, k].item()
                    score = topk_scores[batch_idx, k].item()
                    
                    # Get source beam
                    src_beam = batch_offset + beam_idx
                    prev_tokens = alive_seq[src_beam].tolist()
                    
                    if token_idx == self.eos_id:
                        # Finished hypothesis
                        hyp_tokens = prev_tokens + [token_idx]
                        # Apply length normalization
                        norm_score = score / self.length_norm(len(hyp_tokens))
                        finished[batch_idx].append(
                            BeamHypothesis(tokens=hyp_tokens, score=norm_score)
                        )
                    else:
                        candidates.append((score, src_beam, token_idx))
                    
                    # Stop if we have enough alive beams
                    if len(candidates) >= self.beam_size:
                        break
                
                # Pad candidates if needed
                while len(candidates) < self.beam_size:
                    if candidates:
                        candidates.append(candidates[-1])
                    else:
                        # Fallback: use first beam with pad token
                        candidates.append((float('-inf'), batch_offset, self.pad_id))
                
                # Build new sequences for this batch item
                for score, src_beam, token_idx in candidates[:self.beam_size]:
                    new_seq = torch.cat([
                        alive_seq[src_beam],
                        torch.tensor([token_idx], device=device)
                    ])
                    new_alive_seq.append(new_seq)
                    new_alive_scores.append(score)
            
            # Update alive sequences
            alive_seq = torch.stack(new_alive_seq)
            alive_log_probs = torch.tensor(new_alive_scores, device=device)
            
            # Early stopping: check if best finished > best alive
            all_done = True
            for batch_idx in range(batch_size):
                if finished[batch_idx]:
                    best_finished = max(h.score for h in finished[batch_idx])
                else:
                    best_finished = float('-inf')
                
                batch_offset = batch_idx * self.beam_size
                best_alive = alive_log_probs[batch_offset] / self.length_norm(step + 2)
                
                if best_alive > best_finished:
                    all_done = False
                    break
            
            if all_done:
                break
        
        # Add remaining alive sequences to finished
        for batch_idx in range(batch_size):
            batch_offset = batch_idx * self.beam_size
            for beam in range(self.beam_size):
                idx = batch_offset + beam
                tokens = alive_seq[idx].tolist()
                score = alive_log_probs[idx].item() / self.length_norm(len(tokens))
                finished[batch_idx].append(
                    BeamHypothesis(tokens=tokens, score=score)
                )
        
        # Sort and select top n_best
        results = []
        for batch_idx in range(batch_size):
            sorted_hyps = sorted(finished[batch_idx], key=lambda h: h.score, reverse=True)
            results.append(sorted_hyps[:n_best])
        
        return results


@torch.no_grad()
def beam_search(
    model,
    src: torch.Tensor,
    beam_size: int = 4,
    max_length: int = 256,
    length_penalty: float = 0.6,
    bos_id: int = 2,
    eos_id: int = 3,
    pad_id: int = 0
) -> torch.Tensor:
    """Simple beam search decoding.
    
    Convenience function that returns only the best hypothesis as tensor.
    
    Args:
        model: Transformer model.
        src: Source token IDs of shape (batch, src_len).
        beam_size: Number of beams.
        max_length: Maximum output length.
        length_penalty: Length normalization factor.
        bos_id: BOS token ID.
        eos_id: EOS token ID.
        pad_id: Padding token ID.
    
    Returns:
        Best hypothesis token IDs of shape (batch, out_len).
    """
    decoder = BeamSearchDecoder(
        model=model,
        beam_size=beam_size,
        max_length=max_length,
        length_penalty=length_penalty,
        bos_id=bos_id,
        eos_id=eos_id,
        pad_id=pad_id
    )
    
    results = decoder.decode(src, n_best=1)
    
    # Get best hypothesis for each batch item
    batch_size = src.size(0)
    max_out_len = max(len(results[i][0].tokens) for i in range(batch_size))
    
    output = torch.full(
        (batch_size, max_out_len),
        pad_id,
        dtype=torch.long,
        device=src.device
    )
    
    for i in range(batch_size):
        tokens = results[i][0].tokens
        output[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
    
    return output
