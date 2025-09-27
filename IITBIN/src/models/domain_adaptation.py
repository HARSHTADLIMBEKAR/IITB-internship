"""
Domain adaptation models for cross-modal knowledge transfer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class DomainAdapter(nn.Module):
    """
    Domain adaptation module for transferring knowledge between modalities.
    """
    
    def __init__(
        self,
        source_dim: int,
        target_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        adaptation_method: str = 'linear'  # 'linear', 'mlp', 'residual'
    ):
        """
        Initialize domain adapter.
        
        Args:
            source_dim: Dimension of source modality features
            target_dim: Dimension of target modality features
            hidden_dim: Hidden dimension size
            num_layers: Number of layers in adapter
            dropout: Dropout rate
            adaptation_method: Method for domain adaptation
        """
        super().__init__()
        
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.adaptation_method = adaptation_method
        
        if adaptation_method == 'linear':
            self.adapter = nn.Linear(source_dim, target_dim)
            
        elif adaptation_method == 'mlp':
            layers = []
            current_dim = source_dim
            
            for i in range(num_layers - 1):
                layers.extend([
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                current_dim = hidden_dim
            
            layers.append(nn.Linear(current_dim, target_dim))
            self.adapter = nn.Sequential(*layers)
            
        elif adaptation_method == 'residual':
            self.adapter = nn.Sequential(
                nn.Linear(source_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, target_dim)
            )
            
            # Residual connection
            if source_dim == target_dim:
                self.residual = nn.Identity()
            else:
                self.residual = nn.Linear(source_dim, target_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, source_features: torch.Tensor) -> torch.Tensor:
        """
        Adapt source features to target domain.
        
        Args:
            source_features: Features from source modality
            
        Returns:
            Adapted features for target domain
        """
        if self.adaptation_method == 'residual':
            adapted = self.adapter(source_features)
            residual = self.residual(source_features)
            return adapted + residual
        else:
            return self.adapter(source_features)


class AdversarialDomainAdapter(nn.Module):
    """
    Adversarial domain adaptation for cross-modal transfer.
    """
    
    def __init__(
        self,
        source_dim: int,
        target_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        lambda_adv: float = 1.0
    ):
        """
        Initialize adversarial domain adapter.
        
        Args:
            source_dim: Dimension of source modality features
            target_dim: Dimension of target modality features
            hidden_dim: Hidden dimension size
            num_layers: Number of layers
            dropout: Dropout rate
            lambda_adv: Weight for adversarial loss
        """
        super().__init__()
        
        self.lambda_adv = lambda_adv
        
        # Feature extractor (shared between source and target)
        self.feature_extractor = nn.Sequential(
            nn.Linear(source_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Task classifier
        self.task_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # Binary classification
        )
        
        # Domain classifier (for adversarial training)
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # Source vs Target domain
        )
        
        # Gradient reversal layer
        self.gradient_reversal = GradientReversalLayer(lambda_adv)
        
        # Target domain adapter
        self.target_adapter = nn.Linear(hidden_dim, target_dim)
    
    def forward(
        self, 
        source_features: torch.Tensor,
        target_features: Optional[torch.Tensor] = None,
        return_domain_logits: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through adversarial domain adapter.
        
        Args:
            source_features: Features from source modality
            target_features: Features from target modality (optional)
            return_domain_logits: Whether to return domain classification logits
            
        Returns:
            Dictionary containing adapted features and losses
        """
        # Extract features
        source_extracted = self.feature_extractor(source_features)
        
        # Task classification
        task_logits = self.task_classifier(source_extracted)
        
        # Domain classification (with gradient reversal)
        domain_logits = self.domain_classifier(
            self.gradient_reversal(source_extracted)
        )
        
        # Adapt to target domain
        adapted_features = self.target_adapter(source_extracted)
        
        result = {
            'adapted_features': adapted_features,
            'task_logits': task_logits,
            'source_features': source_extracted
        }
        
        if return_domain_logits:
            result['domain_logits'] = domain_logits
        
        # If target features are provided, compute domain loss
        if target_features is not None:
            target_extracted = self.feature_extractor(target_features)
            target_domain_logits = self.domain_classifier(
                self.gradient_reversal(target_extracted)
            )
            result['target_domain_logits'] = target_domain_logits
        
        return result


class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient reversal layer for adversarial training.
    """
    
    @staticmethod
    def forward(ctx, x, lambda_adv):
        ctx.lambda_adv = lambda_adv
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_adv, None


class ModalityAlignmentNetwork(nn.Module):
    """
    Network for aligning different modalities in a shared space.
    """
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        shared_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize modality alignment network.
        
        Args:
            modality_dims: Dictionary mapping modality names to their dimensions
            shared_dim: Dimension of shared representation space
            num_layers: Number of layers in alignment networks
            dropout: Dropout rate
        """
        super().__init__()
        
        self.modality_dims = modality_dims
        self.shared_dim = shared_dim
        
        # Alignment networks for each modality
        self.alignment_networks = nn.ModuleDict()
        
        for modality, dim in modality_dims.items():
            layers = []
            current_dim = dim
            
            for i in range(num_layers):
                layers.extend([
                    nn.Linear(current_dim, shared_dim if i == num_layers - 1 else shared_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                current_dim = shared_dim
            
            self.alignment_networks[modality] = nn.Sequential(*layers)
        
        # Similarity computation
        self.similarity_net = nn.Sequential(
            nn.Linear(shared_dim * 2, shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self, 
        modality_features: Dict[str, torch.Tensor],
        compute_similarity: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Align modality features to shared space.
        
        Args:
            modality_features: Dictionary of modality features
            compute_similarity: Whether to compute inter-modality similarities
            
        Returns:
            Dictionary containing aligned features and similarities
        """
        aligned_features = {}
        
        # Align each modality
        for modality, features in modality_features.items():
            if modality in self.alignment_networks:
                aligned_features[modality] = self.alignment_networks[modality](features)
        
        result = {'aligned_features': aligned_features}
        
        # Compute similarities if requested
        if compute_similarity and len(aligned_features) > 1:
            similarities = {}
            modalities = list(aligned_features.keys())
            
            for i, mod1 in enumerate(modalities):
                for j, mod2 in enumerate(modalities[i+1:], i+1):
                    # Concatenate features
                    combined = torch.cat([aligned_features[mod1], aligned_features[mod2]], dim=1)
                    similarity = self.similarity_net(combined)
                    similarities[f"{mod1}_{mod2}"] = similarity
            
            result['similarities'] = similarities
        
        return result


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism for knowledge transfer.
    """
    
    def __init__(
        self,
        source_dim: int,
        target_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize cross-modal attention.
        
        Args:
            source_dim: Dimension of source modality
            target_dim: Dimension of target modality
            hidden_dim: Hidden dimension for attention
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.source_proj = nn.Linear(source_dim, hidden_dim)
        self.target_proj = nn.Linear(target_dim, hidden_dim)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.output_proj = nn.Linear(hidden_dim, target_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        source_features: torch.Tensor,
        target_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply cross-modal attention.
        
        Args:
            source_features: Features from source modality
            target_features: Features from target modality
            
        Returns:
            Enhanced target features
        """
        # Project to same dimension
        source_proj = self.source_proj(source_features).unsqueeze(1)
        target_proj = self.target_proj(target_features).unsqueeze(1)
        
        # Cross-attention: target queries, source keys/values
        attn_out, _ = self.attention(target_proj, source_proj, source_proj)
        
        # Project back to target dimension
        enhanced = self.output_proj(attn_out.squeeze(1))
        
        return self.dropout(enhanced + target_features)  # Residual connection
