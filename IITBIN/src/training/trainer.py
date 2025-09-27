"""
Training pipeline for cross-modal knowledge transfer.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path
import json
import time

from .losses import CrossModalLoss, DomainAdaptationLoss, ContrastiveLoss
from .utils import (
    EarlyStopping, ModelCheckpoint, LearningRateScheduler,
    MetricsTracker, GradientClipper, WarmupScheduler, ModelEMA
)
from ..evaluation import ModelEvaluator


class Trainer:
    """
    Base trainer for single-modal models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        device: torch.device,
        config: Dict[str, Any]
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            device: Device to train on
            config: Training configuration
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.config = config
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize utilities
        self.metrics_tracker = MetricsTracker()
        self.gradient_clipper = GradientClipper(
            max_norm=config.get('grad_clip_norm', 1.0)
        )
        
        # Initialize callbacks
        self.early_stopping = EarlyStopping(
            patience=config.get('early_stopping_patience', 10),
            monitor=config.get('monitor_metric', 'val_loss')
        )
        
        self.model_checkpoint = ModelCheckpoint(
            filepath=config.get('checkpoint_path', 'best_model.pth'),
            monitor=config.get('monitor_metric', 'val_loss')
        )
        
        self.lr_scheduler = LearningRateScheduler(
            optimizer=self.optimizer,
            mode='min' if 'loss' in config.get('monitor_metric', 'val_loss') else 'max',
            patience=config.get('lr_patience', 5)
        )
        
        # Move model to device
        self.model.to(device)
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on config."""
        optimizer_name = self.config.get('optimizer', 'adam').lower()
        lr = self.config.get('learning_rate', 1e-3)
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        if optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            momentum = self.config.get('momentum', 0.9)
            return optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = {'train_loss': 0.0, 'train_accuracy': 0.0}
        num_batches = len(self.train_dataloader)
        
        progress_bar = tqdm(self.train_dataloader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            if isinstance(batch, dict):
                data = {k: v.to(self.device) for k, v in batch.items() if k != 'label'}
                targets = batch['label'].to(self.device)
            else:
                data, targets = batch
                data = data.to(self.device)
                targets = targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(data)
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs
            
            # Compute loss
            loss = self.criterion(logits, targets)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            self.gradient_clipper(self.model)
            
            # Update weights
            self.optimizer.step()
            
            # Compute metrics
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == targets).float().mean()
            
            # Update epoch metrics
            epoch_metrics['train_loss'] += loss.item()
            epoch_metrics['train_accuracy'] += accuracy.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{accuracy.item():.4f}"
            })
        
        # Average metrics
        epoch_metrics['train_loss'] /= num_batches
        epoch_metrics['train_accuracy'] /= num_batches
        
        return epoch_metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        epoch_metrics = {'val_loss': 0.0, 'val_accuracy': 0.0}
        num_batches = len(self.val_dataloader)
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                # Move batch to device
                if isinstance(batch, dict):
                    data = {k: v.to(self.device) for k, v in batch.items() if k != 'label'}
                    targets = batch['label'].to(self.device)
                else:
                    data, targets = batch
                    data = data.to(self.device)
                    targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs
                
                # Compute loss
                loss = self.criterion(logits, targets)
                
                # Compute metrics
                predictions = torch.argmax(logits, dim=1)
                accuracy = (predictions == targets).float().mean()
                
                # Update epoch metrics
                epoch_metrics['val_loss'] += loss.item()
                epoch_metrics['val_accuracy'] += accuracy.item()
        
        # Average metrics
        epoch_metrics['val_loss'] /= num_batches
        epoch_metrics['val_accuracy'] /= num_batches
        
        return epoch_metrics
    
    def train(self, num_epochs: int) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs to train
            
        Returns:
            Training history
        """
        print(f"Starting training for {num_epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate_epoch()
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            
            # Update metrics tracker
            self.metrics_tracker.update(epoch_metrics)
            
            # Print metrics
            print(f"Train Loss: {train_metrics['train_loss']:.4f}, "
                  f"Train Acc: {train_metrics['train_accuracy']:.4f}")
            print(f"Val Loss: {val_metrics['val_loss']:.4f}, "
                  f"Val Acc: {val_metrics['val_accuracy']:.4f}")
            
            # Update learning rate
            self.lr_scheduler(val_metrics['val_loss'])
            
            # Check early stopping
            if self.early_stopping(val_metrics['val_loss'], self.model):
                print(f"Early stopping at epoch {epoch + 1}")
                break
            
            # Save checkpoint
            self.model_checkpoint(val_metrics['val_loss'], self.model, self.optimizer, epoch)
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        
        return self.metrics_tracker.history


class CrossModalTrainer:
    """
    Trainer for cross-modal models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        device: torch.device,
        config: Dict[str, Any]
    ):
        """
        Initialize cross-modal trainer.
        
        Args:
            model: Cross-modal model to train
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            device: Device to train on
            config: Training configuration
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.config = config
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize loss function
        self.criterion = CrossModalLoss(
            classification_weight=config.get('classification_weight', 1.0),
            consistency_weight=config.get('consistency_weight', 0.1),
            alignment_weight=config.get('alignment_weight', 0.1)
        )
        
        # Initialize utilities
        self.metrics_tracker = MetricsTracker()
        self.gradient_clipper = GradientClipper(
            max_norm=config.get('grad_clip_norm', 1.0)
        )
        
        # Initialize callbacks
        self.early_stopping = EarlyStopping(
            patience=config.get('early_stopping_patience', 10),
            monitor=config.get('monitor_metric', 'val_total_loss')
        )
        
        self.model_checkpoint = ModelCheckpoint(
            filepath=config.get('checkpoint_path', 'best_cross_modal_model.pth'),
            monitor=config.get('monitor_metric', 'val_total_loss')
        )
        
        self.lr_scheduler = LearningRateScheduler(
            optimizer=self.optimizer,
            mode='min' if 'loss' in config.get('monitor_metric', 'val_total_loss') else 'max',
            patience=config.get('lr_patience', 5)
        )
        
        # Move model to device
        self.model.to(device)
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on config."""
        optimizer_name = self.config.get('optimizer', 'adam').lower()
        lr = self.config.get('learning_rate', 1e-3)
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        if optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            momentum = self.config.get('momentum', 0.9)
            return optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = {
            'train_total_loss': 0.0,
            'train_classification_loss': 0.0,
            'train_consistency_loss': 0.0,
            'train_alignment_loss': 0.0,
            'train_accuracy': 0.0
        }
        num_batches = len(self.train_dataloader)
        
        progress_bar = tqdm(self.train_dataloader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch_data = {k: v.to(self.device) for k, v in batch.items() if k != 'label'}
            targets = batch['label'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(**batch_data, training=True)
            
            # Compute loss
            loss_dict = self.criterion(
                predictions=outputs.get('modality_predictions', {}),
                targets=targets,
                modality_features=outputs.get('modality_features', {})
            )
            
            # Backward pass
            loss_dict['total'].backward()
            
            # Clip gradients
            self.gradient_clipper(self.model)
            
            # Update weights
            self.optimizer.step()
            
            # Compute accuracy
            if 'logits' in outputs:
                predictions = torch.argmax(outputs['logits'], dim=1)
                accuracy = (predictions == targets).float().mean()
            else:
                accuracy = torch.tensor(0.0)
            
            # Update epoch metrics
            for key, value in loss_dict.items():
                epoch_metrics[f'train_{key}_loss'] += value.item()
            epoch_metrics['train_accuracy'] += accuracy.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'total_loss': f"{loss_dict['total'].item():.4f}",
                'acc': f"{accuracy.item():.4f}"
            })
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        epoch_metrics = {
            'val_total_loss': 0.0,
            'val_classification_loss': 0.0,
            'val_consistency_loss': 0.0,
            'val_alignment_loss': 0.0,
            'val_accuracy': 0.0
        }
        num_batches = len(self.val_dataloader)
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                # Move batch to device
                batch_data = {k: v.to(self.device) for k, v in batch.items() if k != 'label'}
                targets = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(**batch_data, training=False)
                
                # Compute loss
                loss_dict = self.criterion(
                    predictions=outputs.get('modality_predictions', {}),
                    targets=targets,
                    modality_features=outputs.get('modality_features', {})
                )
                
                # Compute accuracy
                if 'logits' in outputs:
                    predictions = torch.argmax(outputs['logits'], dim=1)
                    accuracy = (predictions == targets).float().mean()
                else:
                    accuracy = torch.tensor(0.0)
                
                # Update epoch metrics
                for key, value in loss_dict.items():
                    epoch_metrics[f'val_{key}_loss'] += value.item()
                epoch_metrics['val_accuracy'] += accuracy.item()
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    def train(self, num_epochs: int) -> Dict[str, List[float]]:
        """
        Train the cross-modal model.
        
        Args:
            num_epochs: Number of epochs to train
            
        Returns:
            Training history
        """
        print(f"Starting cross-modal training for {num_epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate_epoch()
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            
            # Update metrics tracker
            self.metrics_tracker.update(epoch_metrics)
            
            # Print metrics
            print(f"Train Total Loss: {train_metrics['train_total_loss']:.4f}, "
                  f"Train Acc: {train_metrics['train_accuracy']:.4f}")
            print(f"Val Total Loss: {val_metrics['val_total_loss']:.4f}, "
                  f"Val Acc: {val_metrics['val_accuracy']:.4f}")
            
            # Update learning rate
            self.lr_scheduler(val_metrics['val_total_loss'])
            
            # Check early stopping
            if self.early_stopping(val_metrics['val_total_loss'], self.model):
                print(f"Early stopping at epoch {epoch + 1}")
                break
            
            # Save checkpoint
            self.model_checkpoint(val_metrics['val_total_loss'], self.model, self.optimizer, epoch)
        
        training_time = time.time() - start_time
        print(f"\nCross-modal training completed in {training_time:.2f} seconds")
        
        return self.metrics_tracker.history


class DomainAdaptationTrainer:
    """
    Trainer for domain adaptation models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        source_dataloader: DataLoader,
        target_dataloader: DataLoader,
        device: torch.device,
        config: Dict[str, Any]
    ):
        """
        Initialize domain adaptation trainer.
        
        Args:
            model: Domain adaptation model
            source_dataloader: Source domain data loader
            target_dataloader: Target domain data loader
            device: Device to train on
            config: Training configuration
        """
        self.model = model
        self.source_dataloader = source_dataloader
        self.target_dataloader = target_dataloader
        self.device = device
        self.config = config
        
        # Initialize optimizers
        self.task_optimizer = self._create_optimizer('task')
        self.domain_optimizer = self._create_optimizer('domain')
        
        # Initialize loss function
        self.criterion = DomainAdaptationLoss(
            task_weight=config.get('task_weight', 1.0),
            domain_weight=config.get('domain_weight', 0.1),
            adversarial_weight=config.get('adversarial_weight', 0.1)
        )
        
        # Initialize utilities
        self.metrics_tracker = MetricsTracker()
        self.gradient_clipper = GradientClipper(
            max_norm=config.get('grad_clip_norm', 1.0)
        )
        
        # Move model to device
        self.model.to(device)
    
    def _create_optimizer(self, component: str) -> torch.optim.Optimizer:
        """Create optimizer for specific component."""
        optimizer_name = self.config.get('optimizer', 'adam').lower()
        lr = self.config.get('learning_rate', 1e-3)
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        if component == 'task':
            params = [p for name, p in self.model.named_parameters() 
                     if 'domain' not in name.lower()]
        else:  # domain
            params = [p for name, p in self.model.named_parameters() 
                     if 'domain' in name.lower()]
        
        if optimizer_name == 'adam':
            return optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            momentum = self.config.get('momentum', 0.9)
            return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = {
            'train_task_loss': 0.0,
            'train_domain_loss': 0.0,
            'train_adversarial_loss': 0.0,
            'train_total_loss': 0.0
        }
        
        # Create iterators
        source_iter = iter(self.source_dataloader)
        target_iter = iter(self.target_dataloader)
        
        num_batches = min(len(self.source_dataloader), len(self.target_dataloader))
        progress_bar = tqdm(range(num_batches), desc="Domain Adaptation Training")
        
        for batch_idx in progress_bar:
            try:
                # Get batches
                source_batch = next(source_iter)
                target_batch = next(target_iter)
            except StopIteration:
                # Reset iterators
                source_iter = iter(self.source_dataloader)
                target_iter = iter(self.target_dataloader)
                source_batch = next(source_iter)
                target_batch = next(target_iter)
            
            # Move batches to device
            source_data = {k: v.to(self.device) for k, v in source_batch.items() if k != 'label'}
            source_targets = source_batch['label'].to(self.device)
            
            target_data = {k: v.to(self.device) for k, v in target_batch.items() if k != 'label'}
            target_targets = target_batch['label'].to(self.device)
            
            # Create domain labels (0 for source, 1 for target)
            source_domain_labels = torch.zeros(len(source_targets), dtype=torch.long, device=self.device)
            target_domain_labels = torch.ones(len(target_targets), dtype=torch.long, device=self.device)
            
            # Train task classifier
            self.task_optimizer.zero_grad()
            
            # Source domain forward pass
            source_outputs = self.model(source_data, return_domain_logits=True)
            
            # Task loss
            task_loss = self.criterion.task_loss(
                source_outputs['task_logits'], source_targets
            )
            
            task_loss.backward()
            self.gradient_clipper(self.model)
            self.task_optimizer.step()
            
            # Train domain classifier
            self.domain_optimizer.zero_grad()
            
            # Combined forward pass
            all_data = {k: torch.cat([source_data[k], target_data[k]], dim=0) 
                       for k in source_data.keys()}
            all_domain_labels = torch.cat([source_domain_labels, target_domain_labels], dim=0)
            
            domain_outputs = self.model(all_data, return_domain_logits=True)
            
            # Domain loss
            domain_loss = self.criterion.domain_loss(
                domain_outputs['domain_logits'], all_domain_labels
            )
            
            domain_loss.backward()
            self.gradient_clipper(self.model)
            self.domain_optimizer.step()
            
            # Update metrics
            epoch_metrics['train_task_loss'] += task_loss.item()
            epoch_metrics['train_domain_loss'] += domain_loss.item()
            epoch_metrics['train_total_loss'] += (task_loss + domain_loss).item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'task_loss': f"{task_loss.item():.4f}",
                'domain_loss': f"{domain_loss.item():.4f}"
            })
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    def train(self, num_epochs: int) -> Dict[str, List[float]]:
        """
        Train the domain adaptation model.
        
        Args:
            num_epochs: Number of epochs to train
            
        Returns:
            Training history
        """
        print(f"Starting domain adaptation training for {num_epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch()
            
            # Update metrics tracker
            self.metrics_tracker.update(train_metrics)
            
            # Print metrics
            print(f"Task Loss: {train_metrics['train_task_loss']:.4f}")
            print(f"Domain Loss: {train_metrics['train_domain_loss']:.4f}")
            print(f"Total Loss: {train_metrics['train_total_loss']:.4f}")
        
        training_time = time.time() - start_time
        print(f"\nDomain adaptation training completed in {training_time:.2f} seconds")
        
        return self.metrics_tracker.history
