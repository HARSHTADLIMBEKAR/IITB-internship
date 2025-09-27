"""
Classifier models for multi-modal data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from .base_models import EEGEncoder, EyeTrackingEncoder, GSREncoder, MultiModalEncoder
from .cross_modal import ModalityFusion, ModalityDropout
from .domain_adaptation import DomainAdapter, AdversarialDomainAdapter


class MultiModalClassifier(nn.Module):
    """
    Multi-modal classifier with domain adaptation capabilities.
    """
    
    def __init__(
        self,
        eeg_channels: int = 64,
        eye_tracking_features: int = 4,
        gsr_channels: int = 1,
        sequence_length: int = 1000,
        hidden_dim: int = 128,
        num_classes: int = 2,
        fusion_method: str = 'attention',
        use_domain_adaptation: bool = False,
        modality_dropout: bool = False,
        dropout_prob: float = 0.1
    ):
        """
        Initialize multi-modal classifier.
        
        Args:
            eeg_channels: Number of EEG channels
            eye_tracking_features: Number of eye-tracking features
            gsr_channels: Number of GSR channels
            sequence_length: Length of time sequence
            hidden_dim: Hidden dimension size
            num_classes: Number of output classes
            fusion_method: Method for fusing modalities
            use_domain_adaptation: Whether to use domain adaptation
            modality_dropout: Whether to use modality dropout
            dropout_prob: Dropout rate
        """
        super().__init__()
        
        self.use_domain_adaptation = use_domain_adaptation
        self.modality_dropout = modality_dropout
        
        # Multi-modal encoder
        self.encoder = MultiModalEncoder(
            eeg_channels=eeg_channels,
            eye_tracking_features=eye_tracking_features,
            gsr_channels=gsr_channels,
            sequence_length=sequence_length,
            hidden_dim=hidden_dim,
            fusion_method='concat',  # We'll handle fusion separately
            dropout=dropout_prob
        )
        
        # Modality fusion
        modality_dims = {
            'eeg': self.encoder.eeg_encoder.output_dim,
            'eye_tracking': self.encoder.eye_tracking_encoder.output_dim,
            'gsr': self.encoder.gsr_encoder.output_dim
        }
        
        self.fusion = ModalityFusion(
            modality_dims=modality_dims,
            output_dim=hidden_dim,
            fusion_method=fusion_method
        )
        
        # Modality dropout
        if modality_dropout:
            self.modality_dropout_layer = ModalityDropout(
                dropout_prob=dropout_prob,
                min_modalities=1
            )
        
        # Domain adaptation
        if use_domain_adaptation:
            self.domain_adapter = DomainAdapter(
                source_dim=hidden_dim,
                target_dim=hidden_dim,
                hidden_dim=hidden_dim,
                adaptation_method='mlp'
            )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(
        self,
        eeg: Optional[torch.Tensor] = None,
        eye_tracking: Optional[torch.Tensor] = None,
        gsr: Optional[torch.Tensor] = None,
        training: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through multi-modal classifier.
        
        Args:
            eeg: EEG data
            eye_tracking: Eye-tracking data
            gsr: GSR data
            training: Whether in training mode
            
        Returns:
            Dictionary containing logits and intermediate features
        """
        # Encode each modality
        modality_features = {}
        
        if eeg is not None:
            modality_features['eeg'] = self.encoder.eeg_encoder(eeg)
        
        if eye_tracking is not None:
            modality_features['eye_tracking'] = self.encoder.eye_tracking_encoder(eye_tracking)
        
        if gsr is not None:
            modality_features['gsr'] = self.encoder.gsr_encoder(gsr)
        
        # Apply modality dropout
        if self.modality_dropout and training:
            modality_features = self.modality_dropout_layer(modality_features, training)
        
        # Fuse modalities
        fused_features = self.fusion(modality_features)
        
        # Apply domain adaptation if enabled
        if self.use_domain_adaptation:
            adapted_features = self.domain_adapter(fused_features)
        else:
            adapted_features = fused_features
        
        # Classification
        logits = self.classifier(adapted_features)
        
        return {
            'logits': logits,
            'fused_features': fused_features,
            'modality_features': modality_features,
            'adapted_features': adapted_features
        }


class EEGClassifier(nn.Module):
    """
    EEG-only classifier.
    """
    
    def __init__(
        self,
        input_channels: int = 64,
        sequence_length: int = 1000,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize EEG classifier.
        
        Args:
            input_channels: Number of EEG channels
            sequence_length: Length of time sequence
            hidden_dim: Hidden dimension size
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super().__init__()
        
        self.encoder = EEGEncoder(
            input_channels=input_channels,
            sequence_length=sequence_length,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through EEG classifier.
        
        Args:
            eeg: EEG data
            
        Returns:
            Classification logits
        """
        features = self.encoder(eeg)
        logits = self.classifier(features)
        return logits


class EyeTrackingClassifier(nn.Module):
    """
    Eye-tracking-only classifier.
    """
    
    def __init__(
        self,
        input_features: int = 4,
        sequence_length: int = 1000,
        hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize eye-tracking classifier.
        
        Args:
            input_features: Number of eye-tracking features
            sequence_length: Length of time sequence
            hidden_dim: Hidden dimension size
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super().__init__()
        
        self.encoder = EyeTrackingEncoder(
            input_features=input_features,
            sequence_length=sequence_length,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, eye_tracking: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through eye-tracking classifier.
        
        Args:
            eye_tracking: Eye-tracking data
            
        Returns:
            Classification logits
        """
        features = self.encoder(eye_tracking)
        logits = self.classifier(features)
        return logits


class GSRClassifier(nn.Module):
    """
    GSR-only classifier.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        sequence_length: int = 1000,
        hidden_dim: int = 32,
        num_classes: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize GSR classifier.
        
        Args:
            input_channels: Number of GSR channels
            sequence_length: Length of time sequence
            hidden_dim: Hidden dimension size
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super().__init__()
        
        self.encoder = GSREncoder(
            input_channels=input_channels,
            sequence_length=sequence_length,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, gsr: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GSR classifier.
        
        Args:
            gsr: GSR data
            
        Returns:
            Classification logits
        """
        features = self.encoder(gsr)
        logits = self.classifier(features)
        return logits


class CrossModalTransferClassifier(nn.Module):
    """
    Classifier for cross-modal knowledge transfer.
    """
    
    def __init__(
        self,
        source_encoder: nn.Module,
        target_encoder: nn.Module,
        num_classes: int = 2,
        hidden_dim: int = 128,
        use_adversarial: bool = False,
        dropout: float = 0.1
    ):
        """
        Initialize cross-modal transfer classifier.
        
        Args:
            source_encoder: Encoder for source modality
            target_encoder: Encoder for target modality
            num_classes: Number of output classes
            hidden_dim: Hidden dimension size
            use_adversarial: Whether to use adversarial training
            dropout: Dropout rate
        """
        super().__init__()
        
        self.source_encoder = source_encoder
        self.target_encoder = target_encoder
        self.use_adversarial = use_adversarial
        
        # Domain adaptation
        if use_adversarial:
            self.domain_adapter = AdversarialDomainAdapter(
                source_dim=source_encoder.output_dim,
                target_dim=target_encoder.output_dim,
                hidden_dim=hidden_dim
            )
        else:
            self.domain_adapter = DomainAdapter(
                source_dim=source_encoder.output_dim,
                target_dim=target_encoder.output_dim,
                hidden_dim=hidden_dim
            )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(target_encoder.output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(
        self,
        source_data: torch.Tensor,
        target_data: Optional[torch.Tensor] = None,
        return_domain_logits: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through cross-modal transfer classifier.
        
        Args:
            source_data: Source modality data
            target_data: Target modality data (optional)
            return_domain_logits: Whether to return domain classification logits
            
        Returns:
            Dictionary containing logits and features
        """
        # Encode source data
        source_features = self.source_encoder(source_data)
        
        # Domain adaptation
        if self.use_adversarial:
            adapter_output = self.domain_adapter(
                source_features, 
                target_data=None, 
                return_domain_logits=return_domain_logits
            )
            adapted_features = adapter_output['adapted_features']
            task_logits = adapter_output['task_logits']
        else:
            adapted_features = self.domain_adapter(source_features)
            task_logits = None
        
        # Classification
        logits = self.classifier(adapted_features)
        
        result = {
            'logits': logits,
            'source_features': source_features,
            'adapted_features': adapted_features
        }
        
        if task_logits is not None:
            result['task_logits'] = task_logits
        
        if return_domain_logits and self.use_adversarial:
            result['domain_logits'] = adapter_output['domain_logits']
        
        return result
