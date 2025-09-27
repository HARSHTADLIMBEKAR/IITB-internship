"""
Cross-modal fusion and interaction models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class CrossModalTransformer(nn.Module):
    """
    Transformer-based cross-modal fusion model.
    """
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        fusion_type: str = 'cross_attention'  # 'cross_attention', 'self_attention', 'hybrid'
    ):
        """
        Initialize cross-modal transformer.
        
        Args:
            modality_dims: Dictionary mapping modality names to dimensions
            hidden_dim: Hidden dimension size
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
            fusion_type: Type of fusion mechanism
        """
        super().__init__()
        
        self.modality_dims = modality_dims
        self.hidden_dim = hidden_dim
        self.fusion_type = fusion_type
        
        # Input projections for each modality
        self.input_projections = nn.ModuleDict()
        for modality, dim in modality_dims.items():
            self.input_projections[modality] = nn.Linear(dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Cross-modal attention layers
        if fusion_type in ['cross_attention', 'hybrid']:
            self.cross_attention_layers = nn.ModuleList([
                CrossModalAttentionLayer(hidden_dim, num_heads, dropout)
                for _ in range(num_layers)
            ])
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        modality_features: Dict[str, torch.Tensor],
        modality_masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass through cross-modal transformer.
        
        Args:
            modality_features: Dictionary of modality features
            modality_masks: Optional attention masks for each modality
            
        Returns:
            Fused cross-modal representation
        """
        # Project inputs to hidden dimension
        projected_features = {}
        for modality, features in modality_features.items():
            if modality in self.input_projections:
                projected = self.input_projections[modality](features)
                projected_features[modality] = projected.unsqueeze(1)  # Add sequence dimension
        
        # Combine all modalities
        all_features = torch.cat(list(projected_features.values()), dim=1)
        
        # Add positional encoding
        all_features = self.pos_encoding(all_features)
        
        # Create attention mask
        attention_mask = self._create_attention_mask(
            projected_features, modality_masks
        )
        
        # Apply transformer layers
        if self.fusion_type == 'self_attention':
            # Standard self-attention
            output = self.transformer(all_features, src_key_padding_mask=attention_mask)
        else:
            # Cross-modal attention
            output = all_features
            for i, layer in enumerate(self.transformer.layers):
                # Self-attention
                output = layer(output, src_key_padding_mask=attention_mask)
                
                # Cross-modal attention
                if self.fusion_type in ['cross_attention', 'hybrid']:
                    output = self.cross_attention_layers[i](output, projected_features)
        
        # Global average pooling
        output = torch.mean(output, dim=1)
        
        # Final projection
        output = self.output_projection(output)
        
        return self.dropout(output)
    
    def _create_attention_mask(
        self, 
        projected_features: Dict[str, torch.Tensor],
        modality_masks: Optional[Dict[str, torch.Tensor]]
    ) -> Optional[torch.Tensor]:
        """Create attention mask for transformer."""
        if modality_masks is None:
            return None
        
        # Combine masks from all modalities
        masks = []
        for modality in projected_features.keys():
            if modality in modality_masks:
                masks.append(modality_masks[modality])
            else:
                # No mask for this modality
                batch_size = list(projected_features.values())[0].size(0)
                masks.append(torch.zeros(batch_size, 1, dtype=torch.bool))
        
        return torch.cat(masks, dim=1)


class CrossModalAttentionLayer(nn.Module):
    """Cross-modal attention layer."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        modality_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Apply cross-modal attention."""
        # Residual connection
        residual = x
        
        # Cross-modal attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward
        ff_out = self._feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x
    
    def _feed_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Simple feed-forward network."""
        return F.relu(x)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class ModalityFusion(nn.Module):
    """
    Various fusion strategies for multi-modal data.
    """
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        output_dim: int,
        fusion_method: str = 'concat',  # 'concat', 'add', 'mul', 'attention', 'gated'
        hidden_dim: Optional[int] = None
    ):
        """
        Initialize modality fusion module.
        
        Args:
            modality_dims: Dictionary mapping modality names to dimensions
            output_dim: Output dimension
            fusion_method: Fusion method to use
            hidden_dim: Hidden dimension for complex fusion methods
        """
        super().__init__()
        
        self.modality_dims = modality_dims
        self.fusion_method = fusion_method
        
        if fusion_method == 'concat':
            input_dim = sum(modality_dims.values())
            self.fusion_layer = nn.Linear(input_dim, output_dim)
            
        elif fusion_method == 'add':
            # All modalities must have same dimension
            common_dim = list(modality_dims.values())[0]
            assert all(dim == common_dim for dim in modality_dims.values())
            self.fusion_layer = nn.Linear(common_dim, output_dim)
            
        elif fusion_method == 'mul':
            # All modalities must have same dimension
            common_dim = list(modality_dims.values())[0]
            assert all(dim == common_dim for dim in modality_dims.values())
            self.fusion_layer = nn.Linear(common_dim, output_dim)
            
        elif fusion_method == 'attention':
            if hidden_dim is None:
                hidden_dim = max(modality_dims.values())
            
            # Attention weights for each modality
            self.attention_weights = nn.ModuleDict()
            for modality, dim in modality_dims.items():
                self.attention_weights[modality] = nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, 1)
                )
            
            # Fusion layer
            self.fusion_layer = nn.Linear(max(modality_dims.values()), output_dim)
            
        elif fusion_method == 'gated':
            if hidden_dim is None:
                hidden_dim = max(modality_dims.values())
            
            # Gating mechanism
            self.gates = nn.ModuleDict()
            for modality, dim in modality_dims.items():
                self.gates[modality] = nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.Sigmoid()
                )
            
            # Fusion layer
            self.fusion_layer = nn.Linear(max(modality_dims.values()), output_dim)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, modality_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Fuse modality features.
        
        Args:
            modality_features: Dictionary of modality features
            
        Returns:
            Fused features
        """
        if self.fusion_method == 'concat':
            # Concatenate all features
            fused = torch.cat(list(modality_features.values()), dim=1)
            return self.fusion_layer(fused)
            
        elif self.fusion_method == 'add':
            # Element-wise addition
            fused = torch.stack(list(modality_features.values()), dim=0).sum(dim=0)
            return self.fusion_layer(fused)
            
        elif self.fusion_method == 'mul':
            # Element-wise multiplication
            fused = torch.stack(list(modality_features.values()), dim=0).prod(dim=0)
            return self.fusion_layer(fused)
            
        elif self.fusion_method == 'attention':
            # Attention-weighted fusion
            attention_scores = []
            weighted_features = []
            
            for modality, features in modality_features.items():
                if modality in self.attention_weights:
                    score = self.attention_weights[modality](features)
                    attention_scores.append(score)
                    weighted_features.append(features)
            
            # Softmax attention weights
            attention_weights = F.softmax(torch.cat(attention_scores, dim=1), dim=1)
            
            # Weighted sum
            fused = torch.zeros_like(weighted_features[0])
            for i, features in enumerate(weighted_features):
                fused += attention_weights[:, i:i+1] * features
            
            return self.fusion_layer(fused)
            
        elif self.fusion_method == 'gated':
            # Gated fusion
            gated_features = []
            
            for modality, features in modality_features.items():
                if modality in self.gates:
                    gate = self.gates[modality](features)
                    gated_features.append(gate * features)
            
            # Average gated features
            fused = torch.stack(gated_features, dim=0).mean(dim=0)
            return self.fusion_layer(fused)
        
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")


class ModalityDropout(nn.Module):
    """
    Modality dropout for training robust multi-modal models.
    """
    
    def __init__(
        self,
        dropout_prob: float = 0.3,
        min_modalities: int = 1
    ):
        """
        Initialize modality dropout.
        
        Args:
            dropout_prob: Probability of dropping a modality
            min_modalities: Minimum number of modalities to keep
        """
        super().__init__()
        
        self.dropout_prob = dropout_prob
        self.min_modalities = min_modalities
    
    def forward(
        self, 
        modality_features: Dict[str, torch.Tensor],
        training: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Apply modality dropout.
        
        Args:
            modality_features: Dictionary of modality features
            training: Whether in training mode
            
        Returns:
            Features with some modalities potentially dropped
        """
        if not training or len(modality_features) <= self.min_modalities:
            return modality_features
        
        # Randomly select modalities to drop
        modalities = list(modality_features.keys())
        num_to_drop = max(0, len(modalities) - self.min_modalities)
        
        if num_to_drop > 0:
            drop_mask = torch.rand(len(modalities)) < self.dropout_prob
            # Ensure we don't drop too many
            if drop_mask.sum() > num_to_drop:
                # Keep only the first num_to_drop True values
                true_indices = torch.where(drop_mask)[0]
                drop_mask = torch.zeros_like(drop_mask)
                drop_mask[true_indices[:num_to_drop]] = True
            
            # Apply dropout
            kept_features = {}
            for i, modality in enumerate(modalities):
                if not drop_mask[i]:
                    kept_features[modality] = modality_features[modality]
            
            return kept_features
        
        return modality_features
