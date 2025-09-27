"""
Multi-modal dataset classes for EEG, eye-tracking, and GSR data.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import os
from pathlib import Path


class MultiModalDataset(Dataset):
    """
    Dataset class for multi-modal data (EEG, eye-tracking, GSR).
    Supports domain adaptation and modality dropout.
    """
    
    def __init__(
        self,
        data_path: str,
        modalities: List[str] = ['eeg', 'eye_tracking', 'gsr'],
        target_modality: str = 'eeg',
        domain_adaptation: bool = False,
        modality_dropout: bool = False,
        dropout_prob: float = 0.3,
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None
    ):
        """
        Initialize multi-modal dataset.
        
        Args:
            data_path: Path to the dataset directory
            modalities: List of available modalities
            target_modality: Primary modality for training
            domain_adaptation: Whether to use domain adaptation
            modality_dropout: Whether to apply modality dropout
            dropout_prob: Probability of dropping a modality
            transform: Transform to apply to input data
            target_transform: Transform to apply to target data
        """
        self.data_path = Path(data_path)
        self.modalities = modalities
        self.target_modality = target_modality
        self.domain_adaptation = domain_adaptation
        self.modality_dropout = modality_dropout
        self.dropout_prob = dropout_prob
        self.transform = transform
        self.target_transform = target_transform
        
        # Load data for each modality
        self.data = {}
        self.labels = None
        self._load_data()
        
    def _load_data(self):
        """Load data for all modalities."""
        for modality in self.modalities:
            modality_path = self.data_path / modality
            if modality_path.exists():
                if modality == 'eeg':
                    self.data[modality] = self._load_eeg_data(modality_path)
                elif modality == 'eye_tracking':
                    self.data[modality] = self._load_eye_tracking_data(modality_path)
                elif modality == 'gsr':
                    self.data[modality] = self._load_gsr_data(modality_path)
        
        # Load labels
        labels_path = self.data_path / 'labels.csv'
        if labels_path.exists():
            self.labels = pd.read_csv(labels_path)
        else:
            # Create dummy labels if not provided
            num_samples = len(next(iter(self.data.values())))
            self.labels = pd.DataFrame({
                'label': np.random.randint(0, 2, num_samples),
                'subject_id': np.random.randint(0, 10, num_samples)
            })
    
    def _load_eeg_data(self, path: Path) -> np.ndarray:
        """Load EEG data."""
        # Look for common EEG file formats
        eeg_files = list(path.glob('*.npy')) + list(path.glob('*.npz'))
        if eeg_files:
            return np.load(eeg_files[0])
        else:
            # Create dummy EEG data if no files found
            return np.random.randn(100, 64, 1000)  # 100 samples, 64 channels, 1000 time points
    
    def _load_eye_tracking_data(self, path: Path) -> np.ndarray:
        """Load eye-tracking data."""
        eye_files = list(path.glob('*.npy')) + list(path.glob('*.csv'))
        if eye_files:
            if eye_files[0].suffix == '.csv':
                return pd.read_csv(eye_files[0]).values
            else:
                return np.load(eye_files[0])
        else:
            # Create dummy eye-tracking data
            return np.random.randn(100, 4, 1000)  # 100 samples, 4 features (x, y, pupil_l, pupil_r), 1000 time points
    
    def _load_gsr_data(self, path: Path) -> np.ndarray:
        """Load GSR data."""
        gsr_files = list(path.glob('*.npy')) + list(path.glob('*.csv'))
        if gsr_files:
            if gsr_files[0].suffix == '.csv':
                return pd.read_csv(gsr_files[0]).values
            else:
                return np.load(gsr_files[0])
        else:
            # Create dummy GSR data
            return np.random.randn(100, 1, 1000)  # 100 samples, 1 channel, 1000 time points
    
    def __len__(self) -> int:
        """Return the number of samples."""
        if self.data:
            return len(next(iter(self.data.values())))
        return 0
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing data for each modality and labels
        """
        sample = {}
        
        # Apply modality dropout if enabled
        available_modalities = self.modalities.copy()
        if self.modality_dropout:
            for modality in self.modalities:
                if np.random.random() < self.dropout_prob:
                    available_modalities.remove(modality)
        
        # Load data for available modalities
        for modality in available_modalities:
            if modality in self.data:
                data = self.data[modality][idx]
                if self.transform:
                    data = self.transform(data, modality)
                sample[modality] = torch.tensor(data, dtype=torch.float32)
        
        # Add labels
        if self.labels is not None:
            label = self.labels.iloc[idx]['label']
            if self.target_transform:
                label = self.target_transform(label)
            sample['label'] = torch.tensor(label, dtype=torch.long)
            sample['subject_id'] = torch.tensor(self.labels.iloc[idx]['subject_id'], dtype=torch.long)
        
        return sample


class EEGDataset(Dataset):
    """Dataset class specifically for EEG data."""
    
    def __init__(
        self,
        data_path: str,
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None
    ):
        self.data_path = Path(data_path)
        self.transform = transform
        self.target_transform = target_transform
        self.data, self.labels = self._load_data()
    
    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load EEG data and labels."""
        # Load EEG data
        eeg_files = list(self.data_path.glob('*.npy')) + list(self.data_path.glob('*.npz'))
        if eeg_files:
            data = np.load(eeg_files[0])
        else:
            # Create dummy data
            data = np.random.randn(100, 64, 1000)
        
        # Load labels
        labels_path = self.data_path / 'labels.csv'
        if labels_path.exists():
            labels = pd.read_csv(labels_path)['label'].values
        else:
            labels = np.random.randint(0, 2, len(data))
        
        return data, labels
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
        
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class EyeTrackingDataset(Dataset):
    """Dataset class specifically for eye-tracking data."""
    
    def __init__(
        self,
        data_path: str,
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None
    ):
        self.data_path = Path(data_path)
        self.transform = transform
        self.target_transform = target_transform
        self.data, self.labels = self._load_data()
    
    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load eye-tracking data and labels."""
        # Load eye-tracking data
        eye_files = list(self.data_path.glob('*.npy')) + list(self.data_path.glob('*.csv'))
        if eye_files:
            if eye_files[0].suffix == '.csv':
                data = pd.read_csv(eye_files[0]).values
            else:
                data = np.load(eye_files[0])
        else:
            # Create dummy data
            data = np.random.randn(100, 4, 1000)
        
        # Load labels
        labels_path = self.data_path / 'labels.csv'
        if labels_path.exists():
            labels = pd.read_csv(labels_path)['label'].values
        else:
            labels = np.random.randint(0, 2, len(data))
        
        return data, labels
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
        
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class GSRDataset(Dataset):
    """Dataset class specifically for GSR data."""
    
    def __init__(
        self,
        data_path: str,
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None
    ):
        self.data_path = Path(data_path)
        self.transform = transform
        self.target_transform = target_transform
        self.data, self.labels = self._load_data()
    
    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load GSR data and labels."""
        # Load GSR data
        gsr_files = list(self.data_path.glob('*.npy')) + list(self.data_path.glob('*.csv'))
        if gsr_files:
            if gsr_files[0].suffix == '.csv':
                data = pd.read_csv(gsr_files[0]).values
            else:
                data = np.load(gsr_files[0])
        else:
            # Create dummy data
            data = np.random.randn(100, 1, 1000)
        
        # Load labels
        labels_path = self.data_path / 'labels.csv'
        if labels_path.exists():
            labels = pd.read_csv(labels_path)['label'].values
        else:
            labels = np.random.randint(0, 2, len(data))
        
        return data, labels
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
        
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
