"""
Main Training Loop for NMT.

Production-ready training with:
- Mixed precision (AMP) support
- Gradient accumulation
- Gradient clipping
- Validation and checkpointing
- Early stopping
- Comprehensive logging
"""

import time
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from .scheduler import TransformerScheduler, get_cosine_warmup_scheduler
from .utils import (
    set_seed, save_checkpoint, load_checkpoint,
    EarlyStopping, MetricsTracker, get_device, get_grad_norm
)


class LabelSmoothingLoss(nn.Module):
    """Cross-entropy loss with label smoothing.
    
    Label smoothing prevents the model from becoming overconfident.
    It improves generalization and typically adds 0.5-1.0 BLEU.
    
    Reference:
    - "Rethinking the Inception Architecture for CV" (Szegedy et al., 2016)
    
    Args:
        vocab_size: Vocabulary size.
        padding_idx: Index of padding token (ignored in loss).
        smoothing: Label smoothing factor (0.0 = no smoothing).
    """
    
    def __init__(
        self,
        vocab_size: int,
        padding_idx: int = 0,
        smoothing: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
        # KL divergence loss
        self.criterion = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute label-smoothed cross-entropy loss.
        
        Args:
            logits: Model predictions of shape (batch, seq_len, vocab_size).
            target: Ground truth labels of shape (batch, seq_len).
        
        Returns:
            Scalar loss tensor.
        """
        batch_size, seq_len, vocab_size = logits.size()
        
        # Flatten
        logits = logits.view(-1, vocab_size)  # (batch * seq_len, vocab_size)
        target = target.view(-1)              # (batch * seq_len,)
        
        # Create smoothed distribution
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.smoothing / (vocab_size - 2))  # -2 for target and pad
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist[:, self.padding_idx] = 0
            
            # Mask padding positions
            mask = (target == self.padding_idx)
            true_dist[mask] = 0
        
        # Compute log probabilities and KL divergence
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = self.criterion(log_probs, true_dist)
        
        return loss


class Trainer:
    """Main trainer class for NMT models.
    
    Handles:
    - Training loop with gradient accumulation
    - Validation
    - Mixed precision training
    - Checkpointing
    - Early stopping
    - Metrics logging
    
    Args:
        model: Transformer model.
        config: Training configuration.
        train_dataloader: Training data loader.
        val_dataloader: Validation data loader.
        tokenizer: Tokenizer instance.
        device: Device for training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        tokenizer = None,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.tokenizer = tokenizer
        
        # Device setup
        self.device = device or get_device()
        self.model = self.model.to(self.device)
        
        # Training configuration
        train_cfg = config.training
        model_cfg = config.model
        
        # Loss function with label smoothing
        self.criterion = LabelSmoothingLoss(
            vocab_size=model_cfg.vocab_size,
            padding_idx=model_cfg.pad_id,
            smoothing=train_cfg.label_smoothing
        )
        
        # Optimizer (AdamW with Transformer-specific betas)
        self.optimizer = AdamW(
            model.parameters(),
            lr=train_cfg.learning_rate,
            betas=(train_cfg.adam_beta1, train_cfg.adam_beta2),
            eps=train_cfg.adam_epsilon,
            weight_decay=train_cfg.weight_decay
        )
        
        # Calculate total steps for scheduler
        steps_per_epoch = len(train_dataloader) // train_cfg.gradient_accumulation_steps
        total_steps = steps_per_epoch * train_cfg.max_epochs
        
        # Learning rate scheduler
        self.scheduler = get_cosine_warmup_scheduler(
            self.optimizer,
            warmup_steps=train_cfg.warmup_steps,
            total_steps=total_steps
        )
        
        # Mixed precision scaler
        self.use_amp = train_cfg.use_amp and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Utilities
        self.early_stopping = EarlyStopping(
            patience=train_cfg.patience,
            min_delta=train_cfg.min_delta
        )
        self.metrics_tracker = MetricsTracker(
            log_file=Path(config.model_dir) / "training_log.jsonl"
        )
        
        # Config shortcuts
        self.gradient_accumulation_steps = train_cfg.gradient_accumulation_steps
        self.max_grad_norm = train_cfg.max_grad_norm
        self.save_every_n_steps = train_cfg.save_every_n_steps
        self.validate_every_n_steps = train_cfg.validate_every_n_steps
        
        print(f"Trainer initialized:")
        print(f"  Device: {self.device}")
        print(f"  Mixed precision: {self.use_amp}")
        print(f"  Total parameters: {model.count_parameters_readable()}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary with training metrics.
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.epoch + 1}",
            dynamic_ncols=True
        )
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            src_ids = batch['src_ids'].to(self.device)
            tgt_ids = batch['tgt_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass with AMP
            with autocast(enabled=self.use_amp):
                logits = self.model(src_ids, tgt_ids)
                loss = self.criterion(logits, labels)
                
                # Scale for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update metrics
            epoch_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            
            # Gradient accumulation step
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
                
                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Update progress bar
                current_lr = self.scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    'loss': f"{epoch_loss / num_batches:.4f}",
                    'lr': f"{current_lr:.2e}",
                    'grad': f"{grad_norm:.2f}"
                })
                
                # Track metrics
                self.metrics_tracker.update(
                    loss=epoch_loss / num_batches,
                    learning_rate=current_lr,
                    grad_norm=grad_norm if isinstance(grad_norm, float) else grad_norm.item()
                )
                
                # Periodic validation
                if self.val_dataloader and self.global_step % self.validate_every_n_steps == 0:
                    val_loss = self.validate()
                    self.model.train()
                    
                    # Save best model
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint("best.pt")
                
                # Periodic checkpoint
                if self.global_step % self.save_every_n_steps == 0:
                    self.save_checkpoint(f"step_{self.global_step}.pt")
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        return {'train_loss': avg_loss}
    
    @torch.no_grad()
    def validate(self) -> float:
        """Run validation.
        
        Returns:
            Validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(self.val_dataloader, desc="Validating", leave=False):
            src_ids = batch['src_ids'].to(self.device)
            tgt_ids = batch['tgt_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            with autocast(enabled=self.use_amp):
                logits = self.model(src_ids, tgt_ids)
                loss = self.criterion(logits, labels)
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"\nValidation loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def train(self):
        """Main training loop."""
        print(f"\nStarting training...")
        print(f"  Epochs: {self.config.training.max_epochs}")
        print(f"  Steps per epoch: {len(self.train_dataloader) // self.gradient_accumulation_steps}")
        print(f"  Warmup steps: {self.config.training.warmup_steps}")
        
        start_time = time.time()
        
        for epoch in range(self.config.training.max_epochs):
            self.epoch = epoch
            epoch_start = time.time()
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate
            if self.val_dataloader:
                val_loss = self.validate()
                train_metrics['val_loss'] = val_loss
                
                # Early stopping
                if self.early_stopping(val_loss):
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                    break
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best.pt")
            
            # Log epoch metrics
            epoch_stats = self.metrics_tracker.epoch_end(epoch)
            epoch_time = time.time() - epoch_start
            
            print(f"\nEpoch {epoch + 1} completed in {epoch_time:.1f}s")
            print(f"  Train loss: {train_metrics.get('train_loss', 0):.4f}")
            if 'val_loss' in train_metrics:
                print(f"  Val loss: {train_metrics['val_loss']:.4f}")
            
            # Save epoch checkpoint
            self.save_checkpoint(f"epoch_{epoch + 1}.pt")
        
        # Save final model
        self.save_checkpoint("final.pt")
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time / 3600:.2f} hours")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
    
    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        path = Path(self.config.model_dir) / filename
        save_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.epoch,
            step=self.global_step,
            best_val_loss=self.best_val_loss,
            config=self.model.get_config()
        )
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        meta = load_checkpoint(
            path=Path(path),
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device
        )
        
        self.epoch = meta['epoch']
        self.global_step = meta['step']
        self.best_val_loss = meta['best_val_loss']
