"""
Base model architectures for EEG, eye-tracking, and GSR data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class EEGEncoder(nn.Module):
    """
    Encoder for EEG data using CNN and LSTM architecture.
    """
    
    def __init__(
        self,
        input_channels: int = 64,
        sequence_length: int = 1000,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_attention: bool = True
    ):
        """
        Initialize EEG encoder.
        
        Args:
            input_channels: Number of EEG channels
            sequence_length: Length of time sequence
            hidden_dim: Hidden dimension size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            use_attention: Whether to use attention mechanism
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention
        
        # CNN layers for spatial feature extraction
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim * 2,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
        
        self.dropout = nn.Dropout(dropout)
        
        # Calculate output dimension
        self.output_dim = hidden_dim * 2  # Bidirectional LSTM
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through EEG encoder.
        
        Args:
            x: Input EEG data (batch_size, channels, time)
            
        Returns:
            Encoded representation (batch_size, hidden_dim * 2)
        """
        batch_size = x.size(0)
        
        # CNN feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        # Reshape for LSTM (batch_size, time, features)
        x = x.transpose(1, 2)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention if enabled
        if self.use_attention:
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            # Global average pooling
            output = torch.mean(attn_out, dim=1)
        else:
            # Use last hidden state
            output = hidden[-1]  # Last layer, both directions
        
        return self.dropout(output)


class EyeTrackingEncoder(nn.Module):
    """
    Encoder for eye-tracking data.
    """
    
    def __init__(
        self,
        input_features: int = 4,  # x, y, pupil_l, pupil_r
        sequence_length: int = 1000,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_attention: bool = True
    ):
        """
        Initialize eye-tracking encoder.
        
        Args:
            input_features: Number of input features
            sequence_length: Length of time sequence
            hidden_dim: Hidden dimension size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            use_attention: Whether to use attention mechanism
        """
        super().__init__()
        
        self.input_features = input_features
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention
        
        # Feature projection
        self.feature_proj = nn.Linear(input_features, hidden_dim)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim * 2,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
        
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim * 2
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through eye-tracking encoder.
        
        Args:
            x: Input eye-tracking data (batch_size, features, time)
            
        Returns:
            Encoded representation (batch_size, hidden_dim * 2)
        """
        # Reshape to (batch_size, time, features)
        x = x.transpose(1, 2)
        
        # Feature projection
        x = self.feature_proj(x)
        x = F.relu(x)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention if enabled
        if self.use_attention:
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            # Global average pooling
            output = torch.mean(attn_out, dim=1)
        else:
            # Use last hidden state
            output = hidden[-1]
        
        return self.dropout(output)


class GSREncoder(nn.Module):
    """
    Encoder for GSR data.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        sequence_length: int = 1000,
        hidden_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_attention: bool = True
    ):
        """
        Initialize GSR encoder.
        
        Args:
            input_channels: Number of GSR channels
            sequence_length: Length of time sequence
            hidden_dim: Hidden dimension size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            use_attention: Whether to use attention mechanism
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention
        
        # CNN layers for feature extraction
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(32)
        
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim * 2,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
        
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim * 2
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GSR encoder.
        
        Args:
            x: Input GSR data (batch_size, channels, time)
            
        Returns:
            Encoded representation (batch_size, hidden_dim * 2)
        """
        # CNN feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        # Reshape for LSTM (batch_size, time, features)
        x = x.transpose(1, 2)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention if enabled
        if self.use_attention:
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            # Global average pooling
            output = torch.mean(attn_out, dim=1)
        else:
            # Use last hidden state
            output = hidden[-1]
        
        return self.dropout(output)


class MultiModalEncoder(nn.Module):
    """
    Multi-modal encoder that combines EEG, eye-tracking, and GSR encoders.
    """
    
    def __init__(
        self,
        eeg_channels: int = 64,
        eye_tracking_features: int = 4,
        gsr_channels: int = 1,
        sequence_length: int = 1000,
        hidden_dim: int = 128,
        fusion_method: str = 'concat',  # 'concat', 'attention', 'cross_attention'
        dropout: float = 0.1
    ):
        """
        Initialize multi-modal encoder.
        
        Args:
            eeg_channels: Number of EEG channels
            eye_tracking_features: Number of eye-tracking features
            gsr_channels: Number of GSR channels
            sequence_length: Length of time sequence
            hidden_dim: Hidden dimension size
            fusion_method: Method for fusing modalities
            dropout: Dropout rate
        """
        super().__init__()
        
        self.fusion_method = fusion_method
        
        # Individual encoders
        self.eeg_encoder = EEGEncoder(
            input_channels=eeg_channels,
            sequence_length=sequence_length,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        self.eye_tracking_encoder = EyeTrackingEncoder(
            input_features=eye_tracking_features,
            sequence_length=sequence_length,
            hidden_dim=hidden_dim // 2,
            dropout=dropout
        )
        
        self.gsr_encoder = GSREncoder(
            input_channels=gsr_channels,
            sequence_length=sequence_length,
            hidden_dim=hidden_dim // 4,
            dropout=dropout
        )
        
        # Fusion layers
        if fusion_method == 'concat':
            self.output_dim = (
                self.eeg_encoder.output_dim + 
                self.eye_tracking_encoder.output_dim + 
                self.gsr_encoder.output_dim
            )
            self.fusion_layer = nn.Linear(self.output_dim, hidden_dim)
            
        elif fusion_method == 'attention':
            self.attention_dim = hidden_dim
            self.eeg_proj = nn.Linear(self.eeg_encoder.output_dim, self.attention_dim)
            self.eye_proj = nn.Linear(self.eye_tracking_encoder.output_dim, self.attention_dim)
            self.gsr_proj = nn.Linear(self.gsr_encoder.output_dim, self.attention_dim)
            
            self.attention = nn.MultiheadAttention(
                embed_dim=self.attention_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            
            self.output_dim = self.attention_dim
            
        elif fusion_method == 'cross_attention':
            self.cross_attention_dim = hidden_dim
            self.eeg_proj = nn.Linear(self.eeg_encoder.output_dim, self.cross_attention_dim)
            self.eye_proj = nn.Linear(self.eye_tracking_encoder.output_dim, self.cross_attention_dim)
            self.gsr_proj = nn.Linear(self.gsr_encoder.output_dim, self.cross_attention_dim)
            
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=self.cross_attention_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            
            self.output_dim = self.cross_attention_dim
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        eeg: Optional[torch.Tensor] = None,
        eye_tracking: Optional[torch.Tensor] = None,
        gsr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through multi-modal encoder.
        
        Args:
            eeg: EEG data (batch_size, channels, time)
            eye_tracking: Eye-tracking data (batch_size, features, time)
            gsr: GSR data (batch_size, channels, time)
            
        Returns:
            Fused representation (batch_size, output_dim)
        """
        encoded_features = []
        
        # Encode each modality
        if eeg is not None:
            eeg_encoded = self.eeg_encoder(eeg)
            encoded_features.append(eeg_encoded)
        
        if eye_tracking is not None:
            eye_encoded = self.eye_tracking_encoder(eye_tracking)
            encoded_features.append(eye_encoded)
        
        if gsr is not None:
            gsr_encoded = self.gsr_encoder(gsr)
            encoded_features.append(gsr_encoded)
        
        if not encoded_features:
            raise ValueError("At least one modality must be provided")
        
        # Fuse modalities
        if self.fusion_method == 'concat':
            fused = torch.cat(encoded_features, dim=1)
            output = self.fusion_layer(fused)
            
        elif self.fusion_method == 'attention':
            # Project to same dimension
            projected_features = []
            if eeg is not None:
                projected_features.append(self.eeg_proj(eeg_encoded).unsqueeze(1))
            if eye_tracking is not None:
                projected_features.append(self.eye_proj(eye_encoded).unsqueeze(1))
            if gsr is not None:
                projected_features.append(self.gsr_proj(gsr_encoded).unsqueeze(1))
            
            # Stack for attention
            features_stack = torch.cat(projected_features, dim=1)
            
            # Self-attention
            attn_out, _ = self.attention(features_stack, features_stack, features_stack)
            output = torch.mean(attn_out, dim=1)
            
        elif self.fusion_method == 'cross_attention':
            # Use EEG as query, others as key/value
            if eeg is not None:
                query = self.eeg_proj(eeg_encoded).unsqueeze(1)
                key_value_features = []
                if eye_tracking is not None:
                    key_value_features.append(self.eye_proj(eye_encoded).unsqueeze(1))
                if gsr is not None:
                    key_value_features.append(self.gsr_proj(gsr_encoded).unsqueeze(1))
                
                if key_value_features:
                    key_value = torch.cat(key_value_features, dim=1)
                    attn_out, _ = self.cross_attention(query, key_value, key_value)
                    output = attn_out.squeeze(1)
                else:
                    output = query.squeeze(1)
            else:
                # Fallback to concatenation
                fused = torch.cat(encoded_features, dim=1)
                output = self.fusion_layer(fused)
        
        return self.dropout(output)
