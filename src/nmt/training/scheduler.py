"""
Learning Rate Schedulers for Transformer Training.

Implements:
- Original Transformer schedule (warmup + inverse sqrt decay)
- Linear warmup with cosine decay (alternative)
- Linear warmup with linear decay

Reference:
- "Attention Is All You Need" (Vaswani et al., 2017)
"""

import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class TransformerScheduler(LambdaLR):
    """Original Transformer learning rate schedule.
    
    Formula:
        lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    
    This increases linearly during warmup, then decays as 1/sqrt(step).
    
    Args:
        optimizer: PyTorch optimizer.
        d_model: Model dimension (affects peak learning rate).
        warmup_steps: Number of warmup steps.
        factor: Scaling factor (default 1.0).
        last_epoch: Last epoch for resuming.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        d_model: int,
        warmup_steps: int,
        factor: float = 1.0,
        last_epoch: int = -1
    ):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        
        # Define the LR lambda
        def lr_lambda(step):
            if step == 0:
                step = 1
            return self.factor * (
                self.d_model ** (-0.5) *
                min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
            )
        
        super().__init__(optimizer, lr_lambda, last_epoch)


def get_linear_warmup_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    last_epoch: int = -1
) -> LambdaLR:
    """Linear warmup with linear decay.
    
    Learning rate:
    - Increases linearly from 0 to base_lr during warmup
    - Decreases linearly to 0 after warmup
    
    Args:
        optimizer: PyTorch optimizer.
        warmup_steps: Number of warmup steps.
        total_steps: Total number of training steps.
        last_epoch: Last epoch for resuming.
    
    Returns:
        LambdaLR scheduler.
    """
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return float(step) / float(max(1, warmup_steps))
        else:
            # Linear decay
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 1.0 - progress)
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_warmup_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
    last_epoch: int = -1
) -> LambdaLR:
    """Linear warmup with cosine annealing decay.
    
    Learning rate:
    - Increases linearly from 0 to base_lr during warmup
    - Decays following cosine curve to min_lr_ratio * base_lr
    
    This is a popular alternative to the original Transformer schedule,
    used in many modern NLP models (BERT, GPT, etc.).
    
    Args:
        optimizer: PyTorch optimizer.
        warmup_steps: Number of warmup steps.
        total_steps: Total number of training steps.
        min_lr_ratio: Minimum LR as ratio of base LR.
        last_epoch: Last epoch for resuming.
    
    Returns:
        LambdaLR scheduler.
    """
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return float(step) / float(max(1, warmup_steps))
        else:
            # Cosine decay
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_inverse_sqrt_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    last_epoch: int = -1
) -> LambdaLR:
    """Linear warmup with inverse square root decay.
    
    Simpler version of Transformer schedule without d_model factor.
    
    Learning rate:
    - Increases linearly during warmup
    - Decays as 1/sqrt(step) after warmup
    
    Args:
        optimizer: PyTorch optimizer.
        warmup_steps: Number of warmup steps.
        last_epoch: Last epoch for resuming.
    
    Returns:
        LambdaLR scheduler.
    """
    def lr_lambda(step):
        if step == 0:
            step = 1
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        else:
            return math.sqrt(warmup_steps) / math.sqrt(step)
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


class WarmupConstantScheduler(LambdaLR):
    """Linear warmup followed by constant learning rate.
    
    Useful for fine-tuning or when you want to control LR manually.
    
    Args:
        optimizer: PyTorch optimizer.
        warmup_steps: Number of warmup steps.
        last_epoch: Last epoch for resuming.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        last_epoch: int = -1
    ):
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 1.0
        
        super().__init__(optimizer, lr_lambda, last_epoch)
