"""
Loss functions for cross-modal knowledge transfer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math


class CrossModalLoss(nn.Module):
    """
    Combined loss for cross-modal learning.
    """
    
    def __init__(
        self,
        classification_weight: float = 1.0,
        consistency_weight: float = 0.1,
        alignment_weight: float = 0.1,
        temperature: float = 0.07
    ):
        """
        Initialize cross-modal loss.
        
        Args:
            classification_weight: Weight for classification loss
            consistency_weight: Weight for consistency loss
            alignment_weight: Weight for alignment loss
            temperature: Temperature for contrastive learning
        """
        super().__init__()
        
        self.classification_weight = classification_weight
        self.consistency_weight = consistency_weight
        self.alignment_weight = alignment_weight
        self.temperature = temperature
        
        self.classification_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        modality_features: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute cross-modal loss.
        
        Args:
            predictions: Dictionary of predictions for each modality
            targets: Target labels
            modality_features: Features from each modality
            
        Returns:
            Dictionary containing individual and total losses
        """
        losses = {}
        total_loss = 0.0
        
        # Classification loss for each modality
        classification_losses = []
        for modality, pred in predictions.items():
            if pred is not None:
                loss = self.classification_loss(pred, targets)
                classification_losses.append(loss)
                losses[f'{modality}_classification'] = loss
        
        if classification_losses:
            avg_classification_loss = torch.stack(classification_losses).mean()
            losses['classification'] = avg_classification_loss
            total_loss += self.classification_weight * avg_classification_loss
        
        # Consistency loss (if multiple modalities)
        if len(predictions) > 1 and self.consistency_weight > 0:
            consistency_loss = self._compute_consistency_loss(predictions)
            losses['consistency'] = consistency_loss
            total_loss += self.consistency_weight * consistency_loss
        
        # Alignment loss (if features provided)
        if modality_features is not None and self.alignment_weight > 0:
            alignment_loss = self._compute_alignment_loss(modality_features)
            losses['alignment'] = alignment_loss
            total_loss += self.alignment_weight * alignment_loss
        
        losses['total'] = total_loss
        return losses
    
    def _compute_consistency_loss(self, predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute consistency loss between modalities."""
        modalities = list(predictions.keys())
        consistency_losses = []
        
        for i, mod1 in enumerate(modalities):
            for mod2 in modalities[i+1:]:
                pred1 = F.softmax(predictions[mod1], dim=1)
                pred2 = F.softmax(predictions[mod2], dim=1)
                
                # KL divergence between predictions
                kl_loss = F.kl_div(
                    F.log_softmax(predictions[mod1], dim=1),
                    pred2,
                    reduction='batchmean'
                )
                consistency_losses.append(kl_loss)
        
        return torch.stack(consistency_losses).mean() if consistency_losses else torch.tensor(0.0)
    
    def _compute_alignment_loss(self, modality_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute feature alignment loss."""
        modalities = list(modality_features.keys())
        alignment_losses = []
        
        for i, mod1 in enumerate(modalities):
            for mod2 in modalities[i+1:]:
                features1 = modality_features[mod1]
                features2 = modality_features[mod2]
                
                # Normalize features
                features1 = F.normalize(features1, dim=1)
                features2 = F.normalize(features2, dim=1)
                
                # Cosine similarity loss
                similarity = torch.sum(features1 * features2, dim=1)
                alignment_loss = 1 - torch.mean(similarity)
                alignment_losses.append(alignment_loss)
        
        return torch.stack(alignment_losses).mean() if alignment_losses else torch.tensor(0.0)


class DomainAdaptationLoss(nn.Module):
    """
    Loss for domain adaptation training.
    """
    
    def __init__(
        self,
        task_weight: float = 1.0,
        domain_weight: float = 0.1,
        adversarial_weight: float = 0.1
    ):
        """
        Initialize domain adaptation loss.
        
        Args:
            task_weight: Weight for task classification loss
            domain_weight: Weight for domain classification loss
            adversarial_weight: Weight for adversarial loss
        """
        super().__init__()
        
        self.task_weight = task_weight
        self.domain_weight = domain_weight
        self.adversarial_weight = adversarial_weight
        
        self.task_loss = nn.CrossEntropyLoss()
        self.domain_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        task_logits: torch.Tensor,
        domain_logits: torch.Tensor,
        task_targets: torch.Tensor,
        domain_targets: torch.Tensor,
        adversarial_logits: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute domain adaptation loss.
        
        Args:
            task_logits: Task classification logits
            domain_logits: Domain classification logits
            task_targets: Task labels
            domain_targets: Domain labels
            adversarial_logits: Adversarial classification logits
            
        Returns:
            Dictionary containing individual and total losses
        """
        losses = {}
        
        # Task classification loss
        task_loss = self.task_loss(task_logits, task_targets)
        losses['task'] = task_loss
        
        # Domain classification loss
        domain_loss = self.domain_loss(domain_logits, domain_targets)
        losses['domain'] = domain_loss
        
        # Adversarial loss (if provided)
        if adversarial_logits is not None:
            # Adversarial loss: domain classifier should fail
            adversarial_targets = 1 - domain_targets  # Flip domain labels
            adversarial_loss = self.domain_loss(adversarial_logits, adversarial_targets)
            losses['adversarial'] = adversarial_loss
        
        # Total loss
        total_loss = (
            self.task_weight * task_loss +
            self.domain_weight * domain_loss
        )
        
        if adversarial_logits is not None:
            total_loss += self.adversarial_weight * losses['adversarial']
        
        losses['total'] = total_loss
        return losses


class ContrastiveLoss(nn.Module):
    """
    Contrastive learning loss.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        contrastive_weight: float = 1.0,
        classification_weight: float = 1.0
    ):
        """
        Initialize contrastive loss.
        
        Args:
            temperature: Temperature for contrastive learning
            contrastive_weight: Weight for contrastive loss
            classification_weight: Weight for classification loss
        """
        super().__init__()
        
        self.temperature = temperature
        self.contrastive_weight = contrastive_weight
        self.classification_weight = classification_weight
        
        self.classification_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        classification_logits: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute contrastive loss.
        
        Args:
            features1: First set of features
            features2: Second set of features
            labels: Labels for supervised contrastive learning
            classification_logits: Classification logits
            
        Returns:
            Dictionary containing individual and total losses
        """
        losses = {}
        
        # Contrastive loss
        contrastive_loss = self._compute_contrastive_loss(features1, features2, labels)
        losses['contrastive'] = contrastive_loss
        
        # Classification loss (if provided)
        if classification_logits is not None and labels is not None:
            classification_loss = self.classification_loss(classification_logits, labels)
            losses['classification'] = classification_loss
        
        # Total loss
        total_loss = self.contrastive_weight * contrastive_loss
        
        if classification_logits is not None and labels is not None:
            total_loss += self.classification_weight * losses['classification']
        
        losses['total'] = total_loss
        return losses
    
    def _compute_contrastive_loss(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute contrastive loss between features."""
        # Normalize features
        features1 = F.normalize(features1, dim=1)
        features2 = F.normalize(features2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features1, features2.T) / self.temperature
        
        if labels is not None:
            # Supervised contrastive learning
            return self._supervised_contrastive_loss(similarity_matrix, labels)
        else:
            # Self-supervised contrastive learning
            return self._self_supervised_contrastive_loss(similarity_matrix)
    
    def _supervised_contrastive_loss(
        self,
        similarity_matrix: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute supervised contrastive loss."""
        batch_size = similarity_matrix.size(0)
        
        # Create positive mask
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # Remove diagonal
        mask = mask - torch.eye(batch_size, device=mask.device)
        
        # Compute loss
        exp_sim = torch.exp(similarity_matrix)
        sum_exp_sim = torch.sum(exp_sim, dim=1, keepdim=True)
        
        # Positive pairs
        pos_sim = torch.sum(exp_sim * mask, dim=1)
        
        # Loss
        loss = -torch.log(pos_sim / sum_exp_sim + 1e-8)
        return torch.mean(loss)
    
    def _self_supervised_contrastive_loss(self, similarity_matrix: torch.Tensor) -> torch.Tensor:
        """Compute self-supervised contrastive loss."""
        batch_size = similarity_matrix.size(0)
        labels = torch.arange(batch_size, device=similarity_matrix.device)
        
        return F.cross_entropy(similarity_matrix, labels)


class ModalityDropoutLoss(nn.Module):
    """
    Loss for modality dropout training.
    """
    
    def __init__(
        self,
        classification_weight: float = 1.0,
        reconstruction_weight: float = 0.1,
        regularization_weight: float = 0.01
    ):
        """
        Initialize modality dropout loss.
        
        Args:
            classification_weight: Weight for classification loss
            reconstruction_weight: Weight for reconstruction loss
            regularization_weight: Weight for regularization
        """
        super().__init__()
        
        self.classification_weight = classification_weight
        self.reconstruction_weight = reconstruction_weight
        self.regularization_weight = regularization_weight
        
        self.classification_loss = nn.CrossEntropyLoss()
        self.reconstruction_loss = nn.MSELoss()
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        original_features: Optional[Dict[str, torch.Tensor]] = None,
        reconstructed_features: Optional[Dict[str, torch.Tensor]] = None,
        model_parameters: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute modality dropout loss.
        
        Args:
            predictions: Model predictions
            targets: Target labels
            original_features: Original modality features
            reconstructed_features: Reconstructed modality features
            model_parameters: Model parameters for regularization
            
        Returns:
            Dictionary containing individual and total losses
        """
        losses = {}
        
        # Classification loss
        classification_loss = self.classification_loss(predictions, targets)
        losses['classification'] = classification_loss
        
        # Reconstruction loss (if provided)
        if (original_features is not None and 
            reconstructed_features is not None and 
            self.reconstruction_weight > 0):
            
            reconstruction_losses = []
            for modality in original_features.keys():
                if modality in reconstructed_features:
                    loss = self.reconstruction_loss(
                        reconstructed_features[modality],
                        original_features[modality]
                    )
                    reconstruction_losses.append(loss)
            
            if reconstruction_losses:
                reconstruction_loss = torch.stack(reconstruction_losses).mean()
                losses['reconstruction'] = reconstruction_loss
        
        # Regularization loss (if provided)
        if model_parameters is not None and self.regularization_weight > 0:
            l2_reg = 0.0
            for param in model_parameters:
                l2_reg += torch.norm(param, 2)
            losses['regularization'] = l2_reg
        
        # Total loss
        total_loss = self.classification_weight * classification_loss
        
        if 'reconstruction' in losses:
            total_loss += self.reconstruction_weight * losses['reconstruction']
        
        if 'regularization' in losses:
            total_loss += self.regularization_weight * losses['regularization']
        
        losses['total'] = total_loss
        return losses


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance.
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize focal loss.
        
        Args:
            alpha: Weighting factor
            gamma: Focusing parameter
            reduction: Reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Model predictions
            targets: Target labels
            
        Returns:
            Focal loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
