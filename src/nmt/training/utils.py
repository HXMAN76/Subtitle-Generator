"""
Training Utilities.

Provides:
- Deterministic seeding for reproducibility
- Checkpoint saving and loading
- Logging and metrics tracking
- Early stopping
"""

import os
import random
from pathlib import Path
from typing import Optional, Dict, Any
import json
import numpy as np
import torch
from torch import nn


def set_seed(seed: int = 42, deterministic: bool = False):
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
        deterministic: If True, use fully deterministic operations
                      (may reduce performance by ~10%).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        # For full reproducibility (at some performance cost)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Allow CUDNN to find optimal algorithms (faster, less reproducible)
        torch.backends.cudnn.benchmark = True
    
    # Set environment variable for some operations
    os.environ['PYTHONHASHSEED'] = str(seed)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    step: int = 0,
    best_val_loss: float = float('inf'),
    config: Optional[Dict] = None,
    extra: Optional[Dict] = None
):
    """Save training checkpoint.
    
    Args:
        path: Path to save checkpoint.
        model: Model to save.
        optimizer: Optimizer with state.
        scheduler: Optional learning rate scheduler.
        epoch: Current epoch number.
        step: Current step number.
        best_val_loss: Best validation loss so far.
        config: Model/training configuration.
        extra: Any extra data to save.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'best_val_loss': best_val_loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if config is not None:
        checkpoint['config'] = config
    
    if extra is not None:
        checkpoint.update(extra)
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None
) -> Dict:
    """Load training checkpoint.
    
    Args:
        path: Path to checkpoint file.
        model: Model to load weights into.
        optimizer: Optional optimizer to load state into.
        scheduler: Optional scheduler to load state into.
        device: Device to load tensors to.
    
    Returns:
        Dictionary with checkpoint metadata (epoch, step, etc.).
    """
    if device is None:
        device = torch.device('cpu')
    
    checkpoint = torch.load(path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Checkpoint loaded: {path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Step: {checkpoint.get('step', 'N/A')}")
    print(f"  Best val loss: {checkpoint.get('best_val_loss', 'N/A'):.4f}")
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'step': checkpoint.get('step', 0),
        'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
        'config': checkpoint.get('config'),
    }


class EarlyStopping:
    """Early stopping to halt training when validation loss stops improving.
    
    Args:
        patience: Number of epochs to wait before stopping.
        min_delta: Minimum improvement to consider as progress.
        mode: 'min' for loss, 'max' for metrics like BLEU.
    """
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        mode: str = 'min'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.counter = 0
        self.best_value = None
        self.should_stop = False
    
    def __call__(self, value: float) -> bool:
        """Check if training should stop.
        
        Args:
            value: Current validation metric value.
        
        Returns:
            True if training should stop, False otherwise.
        """
        if self.best_value is None:
            self.best_value = value
            return False
        
        if self.mode == 'min':
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop
    
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_value = None
        self.should_stop = False


class MetricsTracker:
    """Track and log training metrics.
    
    Provides running averages and logging utilities.
    """
    
    def __init__(self, log_file: Optional[Path] = None):
        self.metrics: Dict[str, list] = {}
        self.current_epoch_metrics: Dict[str, list] = {}
        self.log_file = log_file
        
        if log_file is not None:
            log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def update(self, **kwargs):
        """Update metrics with new values."""
        for key, value in kwargs.items():
            if key not in self.current_epoch_metrics:
                self.current_epoch_metrics[key] = []
            
            if isinstance(value, torch.Tensor):
                value = value.item()
            
            self.current_epoch_metrics[key].append(value)
    
    def epoch_end(self, epoch: int):
        """Compute epoch statistics and log."""
        stats = {}
        
        for key, values in self.current_epoch_metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            
            avg_value = sum(values) / len(values) if values else 0
            self.metrics[key].append(avg_value)
            stats[key] = avg_value
        
        # Log to file
        if self.log_file is not None:
            with open(self.log_file, 'a') as f:
                log_entry = {'epoch': epoch, **stats}
                f.write(json.dumps(log_entry) + '\n')
        
        # Reset epoch metrics
        self.current_epoch_metrics = {}
        
        return stats
    
    def get_running_avg(self, key: str, window: int = 100) -> float:
        """Get running average for a metric."""
        if key not in self.current_epoch_metrics:
            return 0.0
        
        values = self.current_epoch_metrics[key][-window:]
        return sum(values) / len(values) if values else 0.0


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(cuda_device: Optional[int] = None) -> torch.device:
    """Get the best available device.
    
    Args:
        cuda_device: Specific CUDA device index, or None for auto.
    
    Returns:
        torch.device for computation.
    """
    if torch.cuda.is_available():
        if cuda_device is not None:
            return torch.device(f'cuda:{cuda_device}')
        return torch.device('cuda')
    return torch.device('cpu')


def get_grad_norm(model: nn.Module, norm_type: float = 2.0) -> float:
    """Compute gradient norm for model parameters."""
    parameters = [p for p in model.parameters() if p.grad is not None]
    if len(parameters) == 0:
        return 0.0
    
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad, norm_type) for p in parameters]),
        norm_type
    )
    return total_norm.item()
