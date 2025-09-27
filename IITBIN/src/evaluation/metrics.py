"""
Evaluation metrics for cross-modal knowledge transfer.
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score
)
from typing import Dict, List, Optional, Tuple, Union
import math


class ClassificationMetrics:
    """
    Standard classification metrics.
    """
    
    def __init__(self, num_classes: int = 2, class_names: Optional[List[str]] = None):
        """
        Initialize classification metrics.
        
        Args:
            num_classes: Number of classes
            class_names: Names of classes
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
        self.probabilities = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, probabilities: Optional[torch.Tensor] = None):
        """
        Update metrics with new predictions.
        
        Args:
            predictions: Predicted class labels
            targets: True class labels
            probabilities: Predicted class probabilities
        """
        self.predictions.extend(predictions.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        
        if probabilities is not None:
            self.probabilities.extend(probabilities.cpu().numpy())
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary of metric values
        """
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(targets, predictions)
        metrics['precision'] = precision_score(targets, predictions, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(targets, predictions, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(targets, predictions, average='weighted', zero_division=0)
        
        # Per-class metrics
        for i in range(self.num_classes):
            class_mask = (targets == i)
            if np.sum(class_mask) > 0:
                class_precision = precision_score(targets, predictions, labels=[i], average='binary', zero_division=0)
                class_recall = recall_score(targets, predictions, labels=[i], average='binary', zero_division=0)
                class_f1 = f1_score(targets, predictions, labels=[i], average='binary', zero_division=0)
                
                metrics[f'{self.class_names[i]}_precision'] = class_precision
                metrics[f'{self.class_names[i]}_recall'] = class_recall
                metrics[f'{self.class_names[i]}_f1'] = class_f1
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        metrics['confusion_matrix'] = cm.tolist()
        
        # ROC AUC (if probabilities available)
        if self.probabilities and len(self.probabilities) > 0:
            probabilities = np.array(self.probabilities)
            if self.num_classes == 2:
                metrics['roc_auc'] = roc_auc_score(targets, probabilities[:, 1])
                metrics['average_precision'] = average_precision_score(targets, probabilities[:, 1])
            else:
                metrics['roc_auc'] = roc_auc_score(targets, probabilities, multi_class='ovr', average='weighted')
                metrics['average_precision'] = average_precision_score(targets, probabilities, average='weighted')
        
        return metrics


class CrossModalMetrics:
    """
    Metrics for cross-modal knowledge transfer evaluation.
    """
    
    def __init__(self, modalities: List[str]):
        """
        Initialize cross-modal metrics.
        
        Args:
            modalities: List of modality names
        """
        self.modalities = modalities
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.modality_predictions = {mod: [] for mod in self.modalities}
        self.modality_targets = {mod: [] for mod in self.modalities}
        self.modality_features = {mod: [] for mod in self.modalities}
    
    def update(
        self, 
        modality_predictions: Dict[str, torch.Tensor],
        modality_targets: Dict[str, torch.Tensor],
        modality_features: Optional[Dict[str, torch.Tensor]] = None
    ):
        """
        Update cross-modal metrics.
        
        Args:
            modality_predictions: Predictions for each modality
            modality_targets: Targets for each modality
            modality_features: Features for each modality
        """
        for modality in self.modalities:
            if modality in modality_predictions:
                self.modality_predictions[modality].extend(
                    modality_predictions[modality].cpu().numpy()
                )
                self.modality_targets[modality].extend(
                    modality_targets[modality].cpu().numpy()
                )
                
                if modality_features and modality in modality_features:
                    self.modality_features[modality].extend(
                        modality_features[modality].cpu().numpy()
                    )
    
    def compute(self) -> Dict[str, Union[float, Dict]]:
        """
        Compute cross-modal metrics.
        
        Returns:
            Dictionary of cross-modal metrics
        """
        metrics = {}
        
        # Individual modality performance
        modality_metrics = {}
        for modality in self.modalities:
            if modality in self.modality_predictions and len(self.modality_predictions[modality]) > 0:
                pred = np.array(self.modality_predictions[modality])
                target = np.array(self.modality_targets[modality])
                
                modality_metrics[modality] = {
                    'accuracy': accuracy_score(target, pred),
                    'precision': precision_score(target, pred, average='weighted', zero_division=0),
                    'recall': recall_score(target, pred, average='weighted', zero_division=0),
                    'f1_score': f1_score(target, pred, average='weighted', zero_division=0)
                }
        
        metrics['modality_performance'] = modality_metrics
        
        # Cross-modal consistency
        if len(self.modalities) > 1:
            consistency_metrics = self._compute_consistency_metrics()
            metrics['consistency'] = consistency_metrics
        
        # Feature similarity (if features available)
        if all(len(self.modality_features[mod]) > 0 for mod in self.modalities):
            similarity_metrics = self._compute_similarity_metrics()
            metrics['feature_similarity'] = similarity_metrics
        
        return metrics
    
    def _compute_consistency_metrics(self) -> Dict[str, float]:
        """Compute cross-modal consistency metrics."""
        consistency_metrics = {}
        
        # Agreement between modalities
        modality_pairs = []
        for i, mod1 in enumerate(self.modalities):
            for mod2 in self.modalities[i+1:]:
                if (mod1 in self.modality_predictions and mod2 in self.modality_predictions and
                    len(self.modality_predictions[mod1]) > 0 and len(self.modality_predictions[mod2]) > 0):
                    
                    pred1 = np.array(self.modality_predictions[mod1])
                    pred2 = np.array(self.modality_predictions[mod2])
                    
                    # Ensure same length
                    min_len = min(len(pred1), len(pred2))
                    pred1 = pred1[:min_len]
                    pred2 = pred2[:min_len]
                    
                    agreement = np.mean(pred1 == pred2)
                    consistency_metrics[f'{mod1}_{mod2}_agreement'] = agreement
        
        return consistency_metrics
    
    def _compute_similarity_metrics(self) -> Dict[str, float]:
        """Compute feature similarity metrics."""
        similarity_metrics = {}
        
        # Cosine similarity between modality features
        modality_pairs = []
        for i, mod1 in enumerate(self.modalities):
            for mod2 in self.modalities[i+1:]:
                if (mod1 in self.modality_features and mod2 in self.modality_features and
                    len(self.modality_features[mod1]) > 0 and len(self.modality_features[mod2]) > 0):
                    
                    features1 = np.array(self.modality_features[mod1])
                    features2 = np.array(self.modality_features[mod2])
                    
                    # Ensure same length
                    min_len = min(len(features1), len(features2))
                    features1 = features1[:min_len]
                    features2 = features2[:min_len]
                    
                    # Compute cosine similarity
                    cosine_sim = self._cosine_similarity(features1, features2)
                    similarity_metrics[f'{mod1}_{mod2}_cosine_similarity'] = cosine_sim
        
        return similarity_metrics
    
    def _cosine_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Compute cosine similarity between feature arrays."""
        # Flatten features if needed
        if features1.ndim > 2:
            features1 = features1.reshape(features1.shape[0], -1)
        if features2.ndim > 2:
            features2 = features2.reshape(features2.shape[0], -1)
        
        # Compute cosine similarity
        dot_product = np.sum(features1 * features2, axis=1)
        norm1 = np.linalg.norm(features1, axis=1)
        norm2 = np.linalg.norm(features2, axis=1)
        
        cosine_sim = dot_product / (norm1 * norm2 + 1e-8)
        return np.mean(cosine_sim)


class DomainAdaptationMetrics:
    """
    Metrics for domain adaptation evaluation.
    """
    
    def __init__(self):
        """Initialize domain adaptation metrics."""
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.source_features = []
        self.target_features = []
        self.adapted_features = []
        self.domain_labels = []  # 0 for source, 1 for target
        self.task_labels = []
    
    def update(
        self,
        source_features: torch.Tensor,
        target_features: Optional[torch.Tensor] = None,
        adapted_features: Optional[torch.Tensor] = None,
        task_labels: Optional[torch.Tensor] = None
    ):
        """
        Update domain adaptation metrics.
        
        Args:
            source_features: Features from source domain
            target_features: Features from target domain
            adapted_features: Adapted features
            task_labels: Task labels
        """
        self.source_features.extend(source_features.cpu().numpy())
        self.domain_labels.extend([0] * len(source_features))
        
        if target_features is not None:
            self.target_features.extend(target_features.cpu().numpy())
            self.domain_labels.extend([1] * len(target_features))
        
        if adapted_features is not None:
            self.adapted_features.extend(adapted_features.cpu().numpy())
        
        if task_labels is not None:
            self.task_labels.extend(task_labels.cpu().numpy())
    
    def compute(self) -> Dict[str, float]:
        """
        Compute domain adaptation metrics.
        
        Returns:
            Dictionary of domain adaptation metrics
        """
        metrics = {}
        
        if len(self.source_features) > 0 and len(self.target_features) > 0:
            # Domain discrepancy
            source_features = np.array(self.source_features)
            target_features = np.array(self.target_features)
            
            # Maximum Mean Discrepancy (MMD)
            mmd = self._compute_mmd(source_features, target_features)
            metrics['mmd'] = mmd
            
            # Wasserstein distance
            wasserstein_dist = self._compute_wasserstein_distance(source_features, target_features)
            metrics['wasserstein_distance'] = wasserstein_dist
        
        if len(self.adapted_features) > 0 and len(self.source_features) > 0:
            # Adaptation quality
            source_features = np.array(self.source_features)
            adapted_features = np.array(self.adapted_features)
            
            # Feature alignment
            alignment_score = self._compute_alignment_score(source_features, adapted_features)
            metrics['alignment_score'] = alignment_score
        
        return metrics
    
    def _compute_mmd(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Compute Maximum Mean Discrepancy."""
        # Simple MMD implementation using RBF kernel
        def rbf_kernel(x, y, gamma=1.0):
            pairwise_dists = np.sum(x**2, axis=1)[:, np.newaxis] + np.sum(y**2, axis=1) - 2 * np.dot(x, y.T)
            return np.exp(-gamma * pairwise_dists)
        
        k11 = rbf_kernel(features1, features1)
        k22 = rbf_kernel(features2, features2)
        k12 = rbf_kernel(features1, features2)
        
        mmd = np.mean(k11) + np.mean(k22) - 2 * np.mean(k12)
        return max(0, mmd)  # MMD is always non-negative
    
    def _compute_wasserstein_distance(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Compute Wasserstein distance (simplified version)."""
        # Simplified Wasserstein distance using L2 norm
        mean1 = np.mean(features1, axis=0)
        mean2 = np.mean(features2, axis=0)
        
        cov1 = np.cov(features1.T)
        cov2 = np.cov(features2.T)
        
        # L2 distance between means
        mean_dist = np.linalg.norm(mean1 - mean2)
        
        # Frobenius norm between covariances
        cov_dist = np.linalg.norm(cov1 - cov2, 'fro')
        
        return mean_dist + cov_dist
    
    def _compute_alignment_score(self, source_features: np.ndarray, adapted_features: np.ndarray) -> float:
        """Compute feature alignment score."""
        # Cosine similarity between source and adapted features
        cosine_sim = self._cosine_similarity(source_features, adapted_features)
        return cosine_sim
    
    def _cosine_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Compute cosine similarity between feature arrays."""
        dot_product = np.sum(features1 * features2, axis=1)
        norm1 = np.linalg.norm(features1, axis=1)
        norm2 = np.linalg.norm(features2, axis=1)
        
        cosine_sim = dot_product / (norm1 * norm2 + 1e-8)
        return np.mean(cosine_sim)


class ContrastiveLearningMetrics:
    """
    Metrics for contrastive learning evaluation.
    """
    
    def __init__(self):
        """Initialize contrastive learning metrics."""
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.positive_pairs = []
        self.negative_pairs = []
        self.embeddings = []
        self.labels = []
    
    def update(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        positive_pairs: Optional[torch.Tensor] = None,
        negative_pairs: Optional[torch.Tensor] = None
    ):
        """
        Update contrastive learning metrics.
        
        Args:
            embeddings: Learned embeddings
            labels: Data labels
            positive_pairs: Positive pair indices
            negative_pairs: Negative pair indices
        """
        self.embeddings.extend(embeddings.cpu().numpy())
        self.labels.extend(labels.cpu().numpy())
        
        if positive_pairs is not None:
            self.positive_pairs.extend(positive_pairs.cpu().numpy())
        
        if negative_pairs is not None:
            self.negative_pairs.extend(negative_pairs.cpu().numpy())
    
    def compute(self) -> Dict[str, float]:
        """
        Compute contrastive learning metrics.
        
        Returns:
            Dictionary of contrastive learning metrics
        """
        metrics = {}
        
        if len(self.embeddings) > 0:
            embeddings = np.array(self.embeddings)
            labels = np.array(self.labels)
            
            # Intra-class and inter-class distances
            intra_class_dist, inter_class_dist = self._compute_class_distances(embeddings, labels)
            metrics['intra_class_distance'] = intra_class_dist
            metrics['inter_class_distance'] = inter_class_dist
            metrics['separation_ratio'] = inter_class_dist / (intra_class_dist + 1e-8)
            
            # Silhouette score
            silhouette_score = self._compute_silhouette_score(embeddings, labels)
            metrics['silhouette_score'] = silhouette_score
        
        return metrics
    
    def _compute_class_distances(self, embeddings: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
        """Compute intra-class and inter-class distances."""
        unique_labels = np.unique(labels)
        
        intra_class_distances = []
        inter_class_distances = []
        
        for label in unique_labels:
            class_mask = (labels == label)
            class_embeddings = embeddings[class_mask]
            
            if len(class_embeddings) > 1:
                # Intra-class distances
                pairwise_distances = np.linalg.norm(
                    class_embeddings[:, np.newaxis] - class_embeddings[np.newaxis, :], axis=2
                )
                # Remove diagonal (self-distances)
                mask = ~np.eye(pairwise_distances.shape[0], dtype=bool)
                intra_class_distances.extend(pairwise_distances[mask])
            
            # Inter-class distances
            other_embeddings = embeddings[~class_mask]
            if len(other_embeddings) > 0:
                inter_distances = np.linalg.norm(
                    class_embeddings[:, np.newaxis] - other_embeddings[np.newaxis, :], axis=2
                )
                inter_class_distances.extend(inter_distances.flatten())
        
        intra_class_dist = np.mean(intra_class_distances) if intra_class_distances else 0
        inter_class_dist = np.mean(inter_class_distances) if inter_class_distances else 0
        
        return intra_class_dist, inter_class_dist
    
    def _compute_silhouette_score(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        """Compute silhouette score."""
        from sklearn.metrics import silhouette_score
        
        if len(np.unique(labels)) < 2:
            return 0.0
        
        return silhouette_score(embeddings, labels)
