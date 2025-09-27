"""
Data transformation modules for multi-modal data.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Union, Tuple
import random


class EEGTransform:
    """Transformations for EEG data."""
    
    def __init__(
        self,
        time_crop: Optional[Tuple[int, int]] = None,
        channel_dropout: float = 0.0,
        noise_std: float = 0.0,
        time_shift: bool = False,
        amplitude_scale: bool = False
    ):
        """
        Initialize EEG transform.
        
        Args:
            time_crop: Tuple of (start, end) for time cropping
            channel_dropout: Probability of dropping channels
            noise_std: Standard deviation of Gaussian noise to add
            time_shift: Whether to apply random time shifts
            amplitude_scale: Whether to apply random amplitude scaling
        """
        self.time_crop = time_crop
        self.channel_dropout = channel_dropout
        self.noise_std = noise_std
        self.time_shift = time_shift
        self.amplitude_scale = amplitude_scale
    
    def __call__(self, eeg_data: torch.Tensor) -> torch.Tensor:
        """Apply transformations to EEG data."""
        # Time cropping
        if self.time_crop:
            start, end = self.time_crop
            eeg_data = eeg_data[:, start:end]
        
        # Channel dropout
        if self.channel_dropout > 0 and random.random() < self.channel_dropout:
            num_channels = eeg_data.shape[0]
            channels_to_drop = random.randint(1, max(1, int(num_channels * 0.1)))
            channels_to_drop = random.sample(range(num_channels), channels_to_drop)
            eeg_data[channels_to_drop] = 0
        
        # Add noise
        if self.noise_std > 0:
            noise = torch.randn_like(eeg_data) * self.noise_std
            eeg_data = eeg_data + noise
        
        # Time shift
        if self.time_shift and eeg_data.shape[1] > 100:
            shift = random.randint(-50, 50)
            eeg_data = torch.roll(eeg_data, shift, dims=1)
        
        # Amplitude scaling
        if self.amplitude_scale:
            scale = random.uniform(0.8, 1.2)
            eeg_data = eeg_data * scale
        
        return eeg_data


class EyeTrackingTransform:
    """Transformations for eye-tracking data."""
    
    def __init__(
        self,
        time_crop: Optional[Tuple[int, int]] = None,
        gaze_noise: float = 0.0,
        pupil_noise: float = 0.0,
        time_shift: bool = False
    ):
        """
        Initialize eye-tracking transform.
        
        Args:
            time_crop: Tuple of (start, end) for time cropping
            gaze_noise: Standard deviation of noise for gaze coordinates
            pupil_noise: Standard deviation of noise for pupil data
            time_shift: Whether to apply random time shifts
        """
        self.time_crop = time_crop
        self.gaze_noise = gaze_noise
        self.pupil_noise = pupil_noise
        self.time_shift = time_shift
    
    def __call__(self, eye_data: torch.Tensor) -> torch.Tensor:
        """Apply transformations to eye-tracking data."""
        # Time cropping
        if self.time_crop:
            start, end = self.time_crop
            eye_data = eye_data[:, start:end]
        
        # Add noise to gaze coordinates (first 2 channels)
        if self.gaze_noise > 0 and eye_data.shape[0] >= 2:
            gaze_noise = torch.randn_like(eye_data[:2]) * self.gaze_noise
            eye_data[:2] = eye_data[:2] + gaze_noise
        
        # Add noise to pupil data (last 2 channels)
        if self.pupil_noise > 0 and eye_data.shape[0] >= 4:
            pupil_noise = torch.randn_like(eye_data[2:4]) * self.pupil_noise
            eye_data[2:4] = eye_data[2:4] + pupil_noise
        
        # Time shift
        if self.time_shift and eye_data.shape[1] > 100:
            shift = random.randint(-50, 50)
            eye_data = torch.roll(eye_data, shift, dims=1)
        
        return eye_data


class GSRTransform:
    """Transformations for GSR data."""
    
    def __init__(
        self,
        time_crop: Optional[Tuple[int, int]] = None,
        noise_std: float = 0.0,
        time_shift: bool = False,
        amplitude_scale: bool = False
    ):
        """
        Initialize GSR transform.
        
        Args:
            time_crop: Tuple of (start, end) for time cropping
            noise_std: Standard deviation of Gaussian noise to add
            time_shift: Whether to apply random time shifts
            amplitude_scale: Whether to apply random amplitude scaling
        """
        self.time_crop = time_crop
        self.noise_std = noise_std
        self.time_shift = time_shift
        self.amplitude_scale = amplitude_scale
    
    def __call__(self, gsr_data: torch.Tensor) -> torch.Tensor:
        """Apply transformations to GSR data."""
        # Time cropping
        if self.time_crop:
            start, end = self.time_crop
            gsr_data = gsr_data[:, start:end]
        
        # Add noise
        if self.noise_std > 0:
            noise = torch.randn_like(gsr_data) * self.noise_std
            gsr_data = gsr_data + noise
        
        # Time shift
        if self.time_shift and gsr_data.shape[1] > 100:
            shift = random.randint(-50, 50)
            gsr_data = torch.roll(gsr_data, shift, dims=1)
        
        # Amplitude scaling
        if self.amplitude_scale:
            scale = random.uniform(0.8, 1.2)
            gsr_data = gsr_data * scale
        
        return gsr_data


class MultiModalTransform:
    """Combined transformations for multi-modal data."""
    
    def __init__(
        self,
        eeg_transform: Optional[EEGTransform] = None,
        eye_tracking_transform: Optional[EyeTrackingTransform] = None,
        gsr_transform: Optional[GSRTransform] = None,
        modality_dropout: float = 0.0
    ):
        """
        Initialize multi-modal transform.
        
        Args:
            eeg_transform: Transform for EEG data
            eye_tracking_transform: Transform for eye-tracking data
            gsr_transform: Transform for GSR data
            modality_dropout: Probability of dropping entire modalities
        """
        self.eeg_transform = eeg_transform
        self.eye_tracking_transform = eye_tracking_transform
        self.gsr_transform = gsr_transform
        self.modality_dropout = modality_dropout
    
    def __call__(self, data: Dict[str, torch.Tensor], modality: str = None) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Apply transformations to multi-modal data.
        
        Args:
            data: Dictionary containing data for each modality or single tensor
            modality: Specific modality to transform (if data is single tensor)
            
        Returns:
            Transformed data
        """
        if isinstance(data, dict):
            # Multi-modal data
            transformed_data = {}
            
            for mod, tensor in data.items():
                if mod == 'eeg' and self.eeg_transform:
                    transformed_data[mod] = self.eeg_transform(tensor)
                elif mod == 'eye_tracking' and self.eye_tracking_transform:
                    transformed_data[mod] = self.eye_tracking_transform(tensor)
                elif mod == 'gsr' and self.gsr_transform:
                    transformed_data[mod] = self.gsr_transform(tensor)
                else:
                    transformed_data[mod] = tensor
            
            # Apply modality dropout
            if self.modality_dropout > 0:
                modalities_to_drop = []
                for mod in transformed_data.keys():
                    if random.random() < self.modality_dropout:
                        modalities_to_drop.append(mod)
                
                for mod in modalities_to_drop:
                    del transformed_data[mod]
            
            return transformed_data
        
        else:
            # Single modality data
            if modality == 'eeg' and self.eeg_transform:
                return self.eeg_transform(data)
            elif modality == 'eye_tracking' and self.eye_tracking_transform:
                return self.eye_tracking_transform(data)
            elif modality == 'gsr' and self.gsr_transform:
                return self.gsr_transform(data)
            else:
                return data


class TimeSeriesAugmentation:
    """Advanced time series augmentation techniques."""
    
    def __init__(
        self,
        jitter_std: float = 0.03,
        scaling_std: float = 0.1,
        rotation_std: float = 0.1,
        permutation_segments: int = 0,
        time_warping: bool = False
    ):
        """
        Initialize time series augmentation.
        
        Args:
            jitter_std: Standard deviation for jittering
            scaling_std: Standard deviation for scaling
            rotation_std: Standard deviation for rotation
            permutation_segments: Number of segments to permute
            time_warping: Whether to apply time warping
        """
        self.jitter_std = jitter_std
        self.scaling_std = scaling_std
        self.rotation_std = rotation_std
        self.permutation_segments = permutation_segments
        self.time_warping = time_warping
    
    def jitter(self, x: torch.Tensor) -> torch.Tensor:
        """Add jittering noise."""
        noise = torch.randn_like(x) * self.jitter_std
        return x + noise
    
    def scaling(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random scaling."""
        scaling_factor = torch.randn(x.shape[0], 1, device=x.device) * self.scaling_std + 1
        return x * scaling_factor
    
    def rotation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random rotation (for 2D data)."""
        if x.shape[0] >= 2:
            angle = torch.randn(1, device=x.device) * self.rotation_std
            cos_a, sin_a = torch.cos(angle), torch.sin(angle)
            rotation_matrix = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], device=x.device)
            
            # Apply rotation to first two channels
            rotated = torch.matmul(rotation_matrix, x[:2])
            result = x.clone()
            result[:2] = rotated
            return result
        return x
    
    def permutation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply segment permutation."""
        if self.permutation_segments > 0:
            seq_len = x.shape[1]
            segment_len = seq_len // self.permutation_segments
            
            if segment_len > 0:
                segments = []
                for i in range(self.permutation_segments):
                    start = i * segment_len
                    end = start + segment_len if i < self.permutation_segments - 1 else seq_len
                    segments.append(x[:, start:end])
                
                # Randomly permute segments
                random.shuffle(segments)
                return torch.cat(segments, dim=1)
        
        return x
    
    def time_warp(self, x: torch.Tensor) -> torch.Tensor:
        """Apply time warping."""
        if self.time_warping:
            seq_len = x.shape[1]
            warp_steps = torch.randint(1, 4, (1,)).item()
            warp_points = torch.sort(torch.randint(0, seq_len, (warp_steps,)))[0]
            
            warped_x = x.clone()
            for i in range(len(warp_points) - 1):
                start, end = warp_points[i], warp_points[i + 1]
                segment_len = end - start
                warp_factor = torch.rand(1) * 0.5 + 0.75  # 0.75 to 1.25
                new_len = int(segment_len * warp_factor)
                
                if new_len > 0:
                    # Resample segment
                    segment = x[:, start:end]
                    resampled = F.interpolate(
                        segment.unsqueeze(0), 
                        size=new_len, 
                        mode='linear', 
                        align_corners=False
                    ).squeeze(0)
                    
                    # Replace original segment
                    if start + new_len <= seq_len:
                        warped_x[:, start:start + new_len] = resampled
                    else:
                        warped_x[:, start:] = resampled[:, :seq_len - start]
            
            return warped_x
        
        return x
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all augmentations."""
        x = self.jitter(x)
        x = self.scaling(x)
        x = self.rotation(x)
        x = self.permutation(x)
        x = self.time_warp(x)
        return x
