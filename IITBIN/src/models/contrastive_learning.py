"""
Contrastive learning models for cross-modal knowledge transfer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class ContrastiveLearner(nn.Module):
    """
    Base contrastive learning module for cross-modal representation learning.
    """
    
    def __init__(
        self,
        encoder_dim: int,
        projection_dim: int = 128,
        temperature: float = 0.07,
        dropout: float = 0.1
    ):
        """
        Initialize contrastive learner.
        
        Args:
            encoder_dim: Dimension of encoder output
            projection_dim: Dimension of projection head
            temperature: Temperature for contrastive loss
            dropout: Dropout rate
        """
        super().__init__()
        
        self.temperature = temperature
        
        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, projection_dim)
        )
        
        # Normalization layer
        self.normalize = nn.LayerNorm(projection_dim)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Project features to contrastive space.
        
        Args:
            features: Input features
            
        Returns:
            Normalized projected features
        """
        projected = self.projection_head(features)
        return F.normalize(self.normalize(projected), dim=1)
    
    def contrastive_loss(
        self, 
        features1: torch.Tensor, 
        features2: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute contrastive loss between two sets of features.
        
        Args:
            features1: First set of features
            features2: Second set of features
            labels: Optional labels for supervised contrastive learning
            
        Returns:
            Contrastive loss
        """
        # Project features
        proj1 = self.forward(features1)
        proj2 = self.forward(features2)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(proj1, proj2.T) / self.temperature
        
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
        
        # Create positive mask (same labels)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # Remove diagonal (self-similarity)
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
        
        # Positive pairs are on the diagonal
        labels = torch.arange(batch_size, device=similarity_matrix.device)
        
        # Cross-entropy loss
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss


class SimCLR(nn.Module):
    """
    SimCLR implementation for cross-modal contrastive learning.
    """
    
    def __init__(
        self,
        encoder_dim: int,
        projection_dim: int = 128,
        temperature: float = 0.07,
        dropout: float = 0.1
    ):
        """
        Initialize SimCLR.
        
        Args:
            encoder_dim: Dimension of encoder output
            projection_dim: Dimension of projection head
            temperature: Temperature for contrastive loss
            dropout: Dropout rate
        """
        super().__init__()
        
        self.contrastive_learner = ContrastiveLearner(
            encoder_dim=encoder_dim,
            projection_dim=projection_dim,
            temperature=temperature,
            dropout=dropout
        )
    
    def forward(
        self, 
        features1: torch.Tensor, 
        features2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute SimCLR loss.
        
        Args:
            features1: First augmented view
            features2: Second augmented view
            
        Returns:
            Contrastive loss
        """
        return self.contrastive_learner.contrastive_loss(features1, features2)


class MoCo(nn.Module):
    """
    Momentum Contrast (MoCo) for cross-modal learning.
    """
    
    def __init__(
        self,
        encoder_dim: int,
        projection_dim: int = 128,
        queue_size: int = 65536,
        momentum: float = 0.999,
        temperature: float = 0.07,
        dropout: float = 0.1
    ):
        """
        Initialize MoCo.
        
        Args:
            encoder_dim: Dimension of encoder output
            projection_dim: Dimension of projection head
            queue_size: Size of the queue for negative samples
            momentum: Momentum for updating key encoder
            temperature: Temperature for contrastive loss
            dropout: Dropout rate
        """
        super().__init__()
        
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature
        
        # Query encoder (main encoder)
        self.query_encoder = ContrastiveLearner(
            encoder_dim=encoder_dim,
            projection_dim=projection_dim,
            temperature=temperature,
            dropout=dropout
        )
        
        # Key encoder (momentum encoder)
        self.key_encoder = ContrastiveLearner(
            encoder_dim=encoder_dim,
            projection_dim=projection_dim,
            temperature=temperature,
            dropout=dropout
        )
        
        # Initialize key encoder with query encoder weights
        self._init_key_encoder()
        
        # Queue for negative samples
        self.register_buffer("queue", torch.randn(projection_dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    def _init_key_encoder(self):
        """Initialize key encoder with query encoder weights."""
        for param_q, param_k in zip(
            self.query_encoder.parameters(), 
            self.key_encoder.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Update key encoder with momentum."""
        for param_q, param_k in zip(
            self.query_encoder.parameters(), 
            self.key_encoder.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """Update the queue with new keys."""
        batch_size = keys.shape[0]
        
        ptr = int(self.queue_ptr)
        
        # Replace the keys at ptr (dequeue and enqueue)
        if ptr + batch_size <= self.queue_size:
            self.queue[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.queue_size
        else:
            # Handle wraparound
            remaining = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:remaining].T
            self.queue[:, :batch_size - remaining] = keys[remaining:].T
            ptr = batch_size - remaining
        
        self.queue_ptr[0] = ptr
    
    def forward(
        self, 
        query_features: torch.Tensor, 
        key_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MoCo loss.
        
        Args:
            query_features: Query features
            key_features: Key features
            
        Returns:
            Contrastive loss
        """
        batch_size = query_features.size(0)
        
        # Encode queries and keys
        q = self.query_encoder(query_features)
        k = self.key_encoder(key_features)
        
        # Detach keys for momentum update
        k = k.detach()
        
        # Positive logits
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        
        # Negative logits
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        
        # Logits
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.temperature
        
        # Labels (positive pairs are at index 0)
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
        
        # Update queue
        self._dequeue_and_enqueue(k)
        
        # Update key encoder
        self._momentum_update_key_encoder()
        
        return F.cross_entropy(logits, labels)


class CrossModalContrastiveLearner(nn.Module):
    """
    Cross-modal contrastive learning for knowledge transfer between modalities.
    """
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        shared_dim: int = 128,
        temperature: float = 0.07,
        dropout: float = 0.1
    ):
        """
        Initialize cross-modal contrastive learner.
        
        Args:
            modality_dims: Dictionary mapping modality names to dimensions
            shared_dim: Dimension of shared representation space
            temperature: Temperature for contrastive loss
            dropout: Dropout rate
        """
        super().__init__()
        
        self.temperature = temperature
        self.modality_dims = modality_dims
        
        # Projection heads for each modality
        self.projection_heads = nn.ModuleDict()
        
        for modality, dim in modality_dims.items():
            self.projection_heads[modality] = nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim, shared_dim)
            )
        
        # Normalization
        self.normalize = nn.LayerNorm(shared_dim)
    
    def forward(self, modality_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Project modality features to shared space.
        
        Args:
            modality_features: Dictionary of modality features
            
        Returns:
            Dictionary of projected features
        """
        projected_features = {}
        
        for modality, features in modality_features.items():
            if modality in self.projection_heads:
                projected = self.projection_heads[modality](features)
                projected_features[modality] = F.normalize(
                    self.normalize(projected), dim=1
                )
        
        return projected_features
    
    def cross_modal_contrastive_loss(
        self, 
        modality_features: Dict[str, torch.Tensor],
        target_modality: str = 'eeg'
    ) -> torch.Tensor:
        """
        Compute cross-modal contrastive loss.
        
        Args:
            modality_features: Dictionary of modality features
            target_modality: Target modality for contrastive learning
            
        Returns:
            Cross-modal contrastive loss
        """
        if target_modality not in modality_features:
            raise ValueError(f"Target modality {target_modality} not found in features")
        
        # Project all features
        projected_features = self.forward(modality_features)
        
        target_features = projected_features[target_modality]
        other_modalities = [mod for mod in projected_features.keys() if mod != target_modality]
        
        if not other_modalities:
            return torch.tensor(0.0, device=target_features.device)
        
        total_loss = 0.0
        
        for modality in other_modalities:
            other_features = projected_features[modality]
            
            # Compute similarity matrix
            similarity_matrix = torch.matmul(target_features, other_features.T) / self.temperature
            
            # Positive pairs are on the diagonal
            batch_size = similarity_matrix.size(0)
            labels = torch.arange(batch_size, device=similarity_matrix.device)
            
            # Cross-entropy loss
            loss = F.cross_entropy(similarity_matrix, labels)
            total_loss += loss
        
        return total_loss / len(other_modalities)
    
    def modality_alignment_loss(
        self, 
        modality_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute modality alignment loss to encourage similar representations.
        
        Args:
            modality_features: Dictionary of modality features
            
        Returns:
            Alignment loss
        """
        projected_features = self.forward(modality_features)
        modalities = list(projected_features.keys())
        
        if len(modalities) < 2:
            return torch.tensor(0.0, device=list(projected_features.values())[0].device)
        
        total_loss = 0.0
        num_pairs = 0
        
        for i, mod1 in enumerate(modalities):
            for mod2 in modalities[i+1:]:
                features1 = projected_features[mod1]
                features2 = projected_features[mod2]
                
                # Cosine similarity loss (maximize similarity)
                similarity = F.cosine_similarity(features1, features2, dim=1)
                loss = 1 - torch.mean(similarity)  # Minimize (1 - similarity)
                
                total_loss += loss
                num_pairs += 1
        
        return total_loss / num_pairs if num_pairs > 0 else torch.tensor(0.0)
