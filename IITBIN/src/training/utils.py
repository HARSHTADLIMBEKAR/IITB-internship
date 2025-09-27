"""
Training utilities for cross-modal knowledge transfer.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
import os
from pathlib import Path
import json
import copy


class EarlyStopping:
    """
    Early stopping utility to prevent overfitting.
    """
    
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        mode: str = 'min',
        restore_best_weights: bool = True,
        monitor: str = 'val_loss'
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
            restore_best_weights: Whether to restore best weights
            monitor: Metric to monitor
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.monitor = monitor
        
        self.wait = 0
        self.stopped_epoch = 0
        self.best_score = None
        self.best_weights = None
        
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater
    
    def __call__(self, current_score: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            current_score: Current metric value
            model: Model to potentially save weights from
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = current_score
            self._save_checkpoint(model)
        elif self.monitor_op(current_score, self.best_score + self.min_delta):
            self.best_score = current_score
            self.wait = 0
            self._save_checkpoint(model)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = self.wait
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        
        return False
    
    def _save_checkpoint(self, model: nn.Module):
        """Save model checkpoint."""
        if self.restore_best_weights:
            self.best_weights = copy.deepcopy(model.state_dict())


class ModelCheckpoint:
    """
    Model checkpoint utility for saving best models.
    """
    
    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        save_weights_only: bool = False,
        verbose: bool = True
    ):
        """
        Initialize model checkpoint.
        
        Args:
            filepath: Path to save model
            monitor: Metric to monitor
            mode: 'min' for loss, 'max' for accuracy
            save_best_only: Whether to save only the best model
            save_weights_only: Whether to save only weights
            verbose: Whether to print messages
        """
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.verbose = verbose
        
        self.best_score = None
        
        if mode == 'min':
            self.monitor_op = np.less
        else:
            self.monitor_op = np.greater
    
    def __call__(self, current_score: float, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int):
        """
        Save model checkpoint if needed.
        
        Args:
            current_score: Current metric value
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
        """
        if self.best_score is None or self.monitor_op(current_score, self.best_score):
            self.best_score = current_score
            
            if self.save_best_only:
                self._save_model(model, optimizer, epoch)
            elif not self.save_best_only:
                self._save_model(model, optimizer, epoch)
    
    def _save_model(self, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int):
        """Save model to file."""
        # Create directory if it doesn't exist
        Path(self.filepath).parent.mkdir(parents=True, exist_ok=True)
        
        if self.save_weights_only:
            torch.save(model.state_dict(), self.filepath)
        else:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_score': self.best_score
            }
            torch.save(checkpoint, self.filepath)
        
        if self.verbose:
            print(f"Model saved to {self.filepath} with score {self.best_score:.4f}")


class LearningRateScheduler:
    """
    Learning rate scheduler with various strategies.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        mode: str = 'min',
        factor: float = 0.1,
        patience: int = 10,
        min_lr: float = 1e-7,
        verbose: bool = True
    ):
        """
        Initialize learning rate scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            mode: 'min' for loss, 'max' for accuracy
            factor: Factor to reduce learning rate
            patience: Number of epochs to wait before reducing
            min_lr: Minimum learning rate
            verbose: Whether to print messages
        """
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        
        self.wait = 0
        self.best_score = None
        
        if mode == 'min':
            self.monitor_op = np.less
        else:
            self.monitor_op = np.greater
    
    def __call__(self, current_score: float):
        """
        Update learning rate if needed.
        
        Args:
            current_score: Current metric value
        """
        if self.best_score is None:
            self.best_score = current_score
        elif self.monitor_op(current_score, self.best_score):
            self.best_score = current_score
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self._reduce_lr()
                self.wait = 0
    
    def _reduce_lr(self):
        """Reduce learning rate."""
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group['lr'] = new_lr
            
            if self.verbose:
                print(f"Reducing learning rate from {old_lr:.6f} to {new_lr:.6f}")


class MetricsTracker:
    """
    Utility for tracking training metrics.
    """
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.metrics = {}
        self.history = {}
    
    def update(self, metrics: Dict[str, float]):
        """
        Update metrics.
        
        Args:
            metrics: Dictionary of metric values
        """
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            
            self.history[key].append(value)
            self.metrics[key] = value
    
    def get_average(self, metric_name: str, window: int = 10) -> float:
        """
        Get average of metric over window.
        
        Args:
            metric_name: Name of metric
            window: Window size for averaging
            
        Returns:
            Average metric value
        """
        if metric_name not in self.history:
            return 0.0
        
        values = self.history[metric_name]
        if len(values) < window:
            return np.mean(values)
        
        return np.mean(values[-window:])
    
    def get_best(self, metric_name: str, mode: str = 'min') -> float:
        """
        Get best value of metric.
        
        Args:
            metric_name: Name of metric
            mode: 'min' or 'max'
            
        Returns:
            Best metric value
        """
        if metric_name not in self.history:
            return 0.0
        
        values = self.history[metric_name]
        if mode == 'min':
            return min(values)
        else:
            return max(values)
    
    def save_history(self, filepath: str):
        """
        Save metrics history to file.
        
        Args:
            filepath: Path to save history
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load_history(self, filepath: str):
        """
        Load metrics history from file.
        
        Args:
            filepath: Path to load history from
        """
        with open(filepath, 'r') as f:
            self.history = json.load(f)


class GradientClipper:
    """
    Gradient clipping utility.
    """
    
    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        """
        Initialize gradient clipper.
        
        Args:
            max_norm: Maximum gradient norm
            norm_type: Type of norm to use
        """
        self.max_norm = max_norm
        self.norm_type = norm_type
    
    def __call__(self, model: nn.Module):
        """
        Clip gradients of model.
        
        Args:
            model: Model to clip gradients for
        """
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm, self.norm_type)


class WarmupScheduler:
    """
    Learning rate warmup scheduler.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int = 5,
        base_lr: float = 1e-3,
        warmup_lr: float = 1e-5
    ):
        """
        Initialize warmup scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            warmup_epochs: Number of warmup epochs
            base_lr: Base learning rate
            warmup_lr: Warmup learning rate
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.warmup_lr = warmup_lr
        self.current_epoch = 0
    
    def step(self):
        """Update learning rate for current epoch."""
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.warmup_lr + (self.base_lr - self.warmup_lr) * self.current_epoch / self.warmup_epochs
        else:
            lr = self.base_lr
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1


class ModelEMA:
    """
    Exponential Moving Average of model parameters.
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        """
        Initialize EMA.
        
        Args:
            model: Model to create EMA for
            decay: EMA decay rate
        """
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model: nn.Module):
        """
        Update EMA parameters.
        
        Args:
            model: Model to update EMA from
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self, model: nn.Module):
        """
        Apply EMA parameters to model.
        
        Args:
            model: Model to apply EMA to
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self, model: nn.Module):
        """
        Restore original parameters.
        
        Args:
            model: Model to restore
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
