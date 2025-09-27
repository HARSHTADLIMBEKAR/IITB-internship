"""
Model evaluation framework for cross-modal knowledge transfer.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from tqdm import tqdm
import json
import os
from pathlib import Path

from .metrics import (
    ClassificationMetrics, CrossModalMetrics, 
    DomainAdaptationMetrics, ContrastiveLearningMetrics
)


class ModelEvaluator:
    """
    General model evaluator for classification tasks.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_classes: int = 2,
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize model evaluator.
        
        Args:
            model: Model to evaluate
            device: Device to run evaluation on
            num_classes: Number of classes
            class_names: Names of classes
        """
        self.model = model
        self.device = device
        self.classification_metrics = ClassificationMetrics(num_classes, class_names)
        
    def evaluate(
        self,
        dataloader: DataLoader,
        return_predictions: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate model on dataset.
        
        Args:
            dataloader: Data loader for evaluation
            return_predictions: Whether to return predictions
            
        Returns:
            Dictionary containing evaluation results
        """
        self.model.eval()
        self.classification_metrics.reset()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Move batch to device
                if isinstance(batch, dict):
                    # Multi-modal data
                    batch_data = {k: v.to(self.device) for k, v in batch.items() if k != 'label'}
                    targets = batch['label'].to(self.device)
                else:
                    # Single modal data
                    batch_data, targets = batch
                    batch_data = batch_data.to(self.device)
                    targets = targets.to(self.device)
                
                # Forward pass
                if isinstance(batch_data, dict):
                    outputs = self.model(**batch_data)
                    if isinstance(outputs, dict):
                        logits = outputs['logits']
                    else:
                        logits = outputs
                else:
                    logits = self.model(batch_data)
                
                # Get predictions
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                # Update metrics
                self.classification_metrics.update(predictions, targets, probabilities)
                
                if return_predictions:
                    all_predictions.extend(predictions.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())
        
        # Compute metrics
        results = self.classification_metrics.compute()
        
        if return_predictions:
            results['predictions'] = all_predictions
            results['targets'] = all_targets
            results['probabilities'] = all_probabilities
        
        return results


class CrossModalEvaluator:
    """
    Evaluator for cross-modal knowledge transfer models.
    """
    
    def __init__(
        self,
        models: Dict[str, nn.Module],
        device: torch.device,
        modalities: List[str],
        num_classes: int = 2,
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize cross-modal evaluator.
        
        Args:
            models: Dictionary of models for each modality
            device: Device to run evaluation on
            modalities: List of modality names
            num_classes: Number of classes
            class_names: Names of classes
        """
        self.models = models
        self.device = device
        self.modalities = modalities
        
        # Initialize metrics
        self.classification_metrics = ClassificationMetrics(num_classes, class_names)
        self.cross_modal_metrics = CrossModalMetrics(modalities)
        self.domain_adaptation_metrics = DomainAdaptationMetrics()
        self.contrastive_metrics = ContrastiveLearningMetrics()
    
    def evaluate_single_modality(
        self,
        modality: str,
        dataloader: DataLoader,
        return_predictions: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate single modality model.
        
        Args:
            modality: Modality name
            dataloader: Data loader for evaluation
            return_predictions: Whether to return predictions
            
        Returns:
            Dictionary containing evaluation results
        """
        if modality not in self.models:
            raise ValueError(f"Model for modality {modality} not found")
        
        model = self.models[modality]
        model.eval()
        
        evaluator = ModelEvaluator(model, self.device)
        return evaluator.evaluate(dataloader, return_predictions)
    
    def evaluate_cross_modal_transfer(
        self,
        source_modality: str,
        target_modality: str,
        source_dataloader: DataLoader,
        target_dataloader: DataLoader,
        transfer_model: Optional[nn.Module] = None
    ) -> Dict[str, Any]:
        """
        Evaluate cross-modal transfer performance.
        
        Args:
            source_modality: Source modality name
            target_modality: Target modality name
            source_dataloader: Source modality data loader
            target_dataloader: Target modality data loader
            transfer_model: Transfer model (optional)
            
        Returns:
            Dictionary containing transfer evaluation results
        """
        results = {}
        
        # Evaluate source modality performance
        source_results = self.evaluate_single_modality(source_modality, source_dataloader)
        results[f'{source_modality}_performance'] = source_results
        
        # Evaluate target modality performance
        target_results = self.evaluate_single_modality(target_modality, target_dataloader)
        results[f'{target_modality}_performance'] = target_results
        
        # Evaluate transfer performance if transfer model provided
        if transfer_model is not None:
            transfer_results = self._evaluate_transfer_model(
                transfer_model, target_dataloader, source_modality, target_modality
            )
            results['transfer_performance'] = transfer_results
        
        # Compute cross-modal metrics
        cross_modal_results = self._compute_cross_modal_metrics(
            source_dataloader, target_dataloader, source_modality, target_modality
        )
        results['cross_modal_metrics'] = cross_modal_results
        
        return results
    
    def _evaluate_transfer_model(
        self,
        transfer_model: nn.Module,
        dataloader: DataLoader,
        source_modality: str,
        target_modality: str
    ) -> Dict[str, Any]:
        """Evaluate transfer model performance."""
        transfer_model.eval()
        self.classification_metrics.reset()
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating transfer model"):
                if isinstance(batch, dict):
                    # Multi-modal data
                    batch_data = {k: v.to(self.device) for k, v in batch.items() if k != 'label'}
                    targets = batch['label'].to(self.device)
                else:
                    batch_data, targets = batch
                    batch_data = batch_data.to(self.device)
                    targets = targets.to(self.device)
                
                # Forward pass through transfer model
                outputs = transfer_model(batch_data)
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs
                
                # Get predictions
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                # Update metrics
                self.classification_metrics.update(predictions, targets, probabilities)
        
        return self.classification_metrics.compute()
    
    def _compute_cross_modal_metrics(
        self,
        source_dataloader: DataLoader,
        target_dataloader: DataLoader,
        source_modality: str,
        target_modality: str
    ) -> Dict[str, Any]:
        """Compute cross-modal metrics."""
        self.cross_modal_metrics.reset()
        
        # Collect predictions and features from both modalities
        source_predictions = []
        target_predictions = []
        source_features = []
        target_features = []
        
        # Source modality
        source_model = self.models[source_modality]
        source_model.eval()
        
        with torch.no_grad():
            for batch in tqdm(source_dataloader, desc="Collecting source features"):
                if isinstance(batch, dict):
                    batch_data = {k: v.to(self.device) for k, v in batch.items() if k != 'label'}
                    targets = batch['label'].to(self.device)
                else:
                    batch_data, targets = batch
                    batch_data = batch_data.to(self.device)
                    targets = targets.to(self.device)
                
                # Get predictions and features
                if hasattr(source_model, 'encoder'):
                    features = source_model.encoder(batch_data)
                    logits = source_model.classifier(features)
                else:
                    logits = source_model(batch_data)
                    features = None
                
                predictions = torch.argmax(logits, dim=1)
                source_predictions.append(predictions.cpu())
                
                if features is not None:
                    source_features.append(features.cpu())
        
        # Target modality
        target_model = self.models[target_modality]
        target_model.eval()
        
        with torch.no_grad():
            for batch in tqdm(target_dataloader, desc="Collecting target features"):
                if isinstance(batch, dict):
                    batch_data = {k: v.to(self.device) for k, v in batch.items() if k != 'label'}
                    targets = batch['label'].to(self.device)
                else:
                    batch_data, targets = batch
                    batch_data = batch_data.to(self.device)
                    targets = targets.to(self.device)
                
                # Get predictions and features
                if hasattr(target_model, 'encoder'):
                    features = target_model.encoder(batch_data)
                    logits = target_model.classifier(features)
                else:
                    logits = target_model(batch_data)
                    features = None
                
                predictions = torch.argmax(logits, dim=1)
                target_predictions.append(predictions.cpu())
                
                if features is not None:
                    target_features.append(features.cpu())
        
        # Update cross-modal metrics
        modality_predictions = {
            source_modality: torch.cat(source_predictions),
            target_modality: torch.cat(target_predictions)
        }
        
        modality_targets = {
            source_modality: torch.cat(source_predictions),  # Using predictions as targets for consistency
            target_modality: torch.cat(target_predictions)
        }
        
        modality_features = {}
        if source_features:
            modality_features[source_modality] = torch.cat(source_features)
        if target_features:
            modality_features[target_modality] = torch.cat(target_features)
        
        self.cross_modal_metrics.update(
            modality_predictions, modality_targets, modality_features
        )
        
        return self.cross_modal_metrics.compute()
    
    def evaluate_domain_adaptation(
        self,
        source_dataloader: DataLoader,
        target_dataloader: DataLoader,
        adaptation_model: nn.Module
    ) -> Dict[str, Any]:
        """
        Evaluate domain adaptation performance.
        
        Args:
            source_dataloader: Source domain data loader
            target_dataloader: Target domain data loader
            adaptation_model: Domain adaptation model
            
        Returns:
            Dictionary containing domain adaptation evaluation results
        """
        adaptation_model.eval()
        self.domain_adaptation_metrics.reset()
        
        # Collect features from both domains
        with torch.no_grad():
            # Source domain
            for batch in tqdm(source_dataloader, desc="Collecting source domain features"):
                if isinstance(batch, dict):
                    batch_data = {k: v.to(self.device) for k, v in batch.items() if k != 'label'}
                    targets = batch['label'].to(self.device)
                else:
                    batch_data, targets = batch
                    batch_data = batch_data.to(self.device)
                    targets = targets.to(self.device)
                
                # Get source features
                if hasattr(adaptation_model, 'source_encoder'):
                    source_features = adaptation_model.source_encoder(batch_data)
                else:
                    source_features = adaptation_model(batch_data)
                
                self.domain_adaptation_metrics.update(
                    source_features, None, None, targets
                )
            
            # Target domain
            for batch in tqdm(target_dataloader, desc="Collecting target domain features"):
                if isinstance(batch, dict):
                    batch_data = {k: v.to(self.device) for k, v in batch.items() if k != 'label'}
                    targets = batch['label'].to(self.device)
                else:
                    batch_data, targets = batch
                    batch_data = batch_data.to(self.device)
                    targets = targets.to(self.device)
                
                # Get target features
                if hasattr(adaptation_model, 'target_encoder'):
                    target_features = adaptation_model.target_encoder(batch_data)
                else:
                    target_features = adaptation_model(batch_data)
                
                self.domain_adaptation_metrics.update(
                    None, target_features, None, targets
                )
        
        return self.domain_adaptation_metrics.compute()
    
    def save_results(self, results: Dict[str, Any], save_path: str):
        """
        Save evaluation results to file.
        
        Args:
            results: Evaluation results
            save_path: Path to save results
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        results_serializable = convert_numpy(results)
        
        with open(save_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"Results saved to {save_path}")
    
    def load_results(self, load_path: str) -> Dict[str, Any]:
        """
        Load evaluation results from file.
        
        Args:
            load_path: Path to load results from
            
        Returns:
            Loaded evaluation results
        """
        with open(load_path, 'r') as f:
            results = json.load(f)
        
        return results
