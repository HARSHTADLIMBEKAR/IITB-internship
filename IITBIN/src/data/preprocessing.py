"""
Data preprocessing modules for EEG, eye-tracking, and GSR data.
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Dict, List, Optional, Tuple, Union
import mne
from mne.filter import filter_data


class EEGPreprocessor:
    """
    Preprocessing pipeline for EEG data.
    """
    
    def __init__(
        self,
        sampling_rate: int = 1000,
        low_freq: float = 1.0,
        high_freq: float = 40.0,
        notch_freq: float = 50.0,
        normalize: bool = True,
        remove_artifacts: bool = True
    ):
        """
        Initialize EEG preprocessor.
        
        Args:
            sampling_rate: Sampling rate of EEG data
            low_freq: Low frequency cutoff for bandpass filter
            high_freq: High frequency cutoff for bandpass filter
            notch_freq: Notch filter frequency (for power line noise)
            normalize: Whether to normalize the data
            remove_artifacts: Whether to remove artifacts
        """
        self.sampling_rate = sampling_rate
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.notch_freq = notch_freq
        self.normalize = normalize
        self.remove_artifacts = remove_artifacts
        self.scaler = StandardScaler() if normalize else None
    
    def preprocess(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Apply full preprocessing pipeline to EEG data.
        
        Args:
            eeg_data: Raw EEG data (channels x time_points)
            
        Returns:
            Preprocessed EEG data
        """
        # Ensure data is in correct format (channels x time)
        if eeg_data.ndim == 3:
            # If batch dimension exists, process each sample
            processed_data = np.zeros_like(eeg_data)
            for i in range(eeg_data.shape[0]):
                processed_data[i] = self._preprocess_single(eeg_data[i])
            return processed_data
        else:
            return self._preprocess_single(eeg_data)
    
    def _preprocess_single(self, eeg_data: np.ndarray) -> np.ndarray:
        """Preprocess a single EEG sample."""
        # Remove DC offset
        eeg_data = eeg_data - np.mean(eeg_data, axis=1, keepdims=True)
        
        # Apply notch filter (remove power line noise)
        if self.notch_freq:
            eeg_data = self._apply_notch_filter(eeg_data)
        
        # Apply bandpass filter
        eeg_data = self._apply_bandpass_filter(eeg_data)
        
        # Remove artifacts (simple threshold-based)
        if self.remove_artifacts:
            eeg_data = self._remove_artifacts(eeg_data)
        
        # Normalize
        if self.normalize:
            eeg_data = self._normalize(eeg_data)
        
        return eeg_data
    
    def _apply_notch_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply notch filter to remove power line noise."""
        nyquist = self.sampling_rate / 2
        low = (self.notch_freq - 1) / nyquist
        high = (self.notch_freq + 1) / nyquist
        b, a = signal.butter(4, [low, high], btype='bandstop')
        
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered_data[i] = signal.filtfilt(b, a, data[i])
        
        return filtered_data
    
    def _apply_bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply bandpass filter."""
        nyquist = self.sampling_rate / 2
        low = self.low_freq / nyquist
        high = self.high_freq / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered_data[i] = signal.filtfilt(b, a, data[i])
        
        return filtered_data
    
    def _remove_artifacts(self, data: np.ndarray) -> np.ndarray:
        """Remove artifacts using threshold-based method."""
        # Simple artifact removal: clip extreme values
        threshold = 3 * np.std(data)
        data = np.clip(data, -threshold, threshold)
        return data
    
    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize data using z-score."""
        return zscore(data, axis=1)


class EyeTrackingPreprocessor:
    """
    Preprocessing pipeline for eye-tracking data.
    """
    
    def __init__(
        self,
        sampling_rate: int = 1000,
        smooth_window: int = 5,
        normalize: bool = True,
        remove_blinks: bool = True
    ):
        """
        Initialize eye-tracking preprocessor.
        
        Args:
            sampling_rate: Sampling rate of eye-tracking data
            smooth_window: Window size for smoothing
            normalize: Whether to normalize the data
            remove_blinks: Whether to remove blink artifacts
        """
        self.sampling_rate = sampling_rate
        self.smooth_window = smooth_window
        self.normalize = normalize
        self.remove_blinks = remove_blinks
        self.scaler = StandardScaler() if normalize else None
    
    def preprocess(self, eye_data: np.ndarray) -> np.ndarray:
        """
        Apply full preprocessing pipeline to eye-tracking data.
        
        Args:
            eye_data: Raw eye-tracking data (features x time_points)
            
        Returns:
            Preprocessed eye-tracking data
        """
        if eye_data.ndim == 3:
            processed_data = np.zeros_like(eye_data)
            for i in range(eye_data.shape[0]):
                processed_data[i] = self._preprocess_single(eye_data[i])
            return processed_data
        else:
            return self._preprocess_single(eye_data)
    
    def _preprocess_single(self, eye_data: np.ndarray) -> np.ndarray:
        """Preprocess a single eye-tracking sample."""
        # Remove blinks
        if self.remove_blinks:
            eye_data = self._remove_blinks(eye_data)
        
        # Smooth the data
        eye_data = self._smooth_data(eye_data)
        
        # Normalize
        if self.normalize:
            eye_data = self._normalize(eye_data)
        
        return eye_data
    
    def _remove_blinks(self, data: np.ndarray) -> np.ndarray:
        """Remove blink artifacts."""
        # Simple blink detection based on pupil diameter
        if data.shape[0] >= 3:  # Assuming pupil data is in last two channels
            pupil_data = data[-2:]  # Left and right pupil
            blink_threshold = np.mean(pupil_data) - 2 * np.std(pupil_data)
            
            # Interpolate blink periods
            for i in range(pupil_data.shape[0]):
                blink_mask = pupil_data[i] < blink_threshold
                if np.any(blink_mask):
                    # Linear interpolation for blink periods
                    valid_indices = ~blink_mask
                    if np.sum(valid_indices) > 1:
                        data[i] = np.interp(
                            np.arange(len(data[i])),
                            np.where(valid_indices)[0],
                            data[i][valid_indices]
                        )
        
        return data
    
    def _smooth_data(self, data: np.ndarray) -> np.ndarray:
        """Apply smoothing to reduce noise."""
        if self.smooth_window > 1:
            smoothed_data = np.zeros_like(data)
            for i in range(data.shape[0]):
                smoothed_data[i] = signal.savgol_filter(
                    data[i], 
                    self.smooth_window, 
                    3 if self.smooth_window >= 3 else 1
                )
            return smoothed_data
        return data
    
    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize data."""
        return zscore(data, axis=1)


class GSRPreprocessor:
    """
    Preprocessing pipeline for GSR data.
    """
    
    def __init__(
        self,
        sampling_rate: int = 1000,
        low_freq: float = 0.05,
        high_freq: float = 2.0,
        normalize: bool = True,
        smooth_window: int = 10
    ):
        """
        Initialize GSR preprocessor.
        
        Args:
            sampling_rate: Sampling rate of GSR data
            low_freq: Low frequency cutoff for bandpass filter
            high_freq: High frequency cutoff for bandpass filter
            normalize: Whether to normalize the data
            smooth_window: Window size for smoothing
        """
        self.sampling_rate = sampling_rate
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.normalize = normalize
        self.smooth_window = smooth_window
        self.scaler = StandardScaler() if normalize else None
    
    def preprocess(self, gsr_data: np.ndarray) -> np.ndarray:
        """
        Apply full preprocessing pipeline to GSR data.
        
        Args:
            gsr_data: Raw GSR data (channels x time_points)
            
        Returns:
            Preprocessed GSR data
        """
        if gsr_data.ndim == 3:
            processed_data = np.zeros_like(gsr_data)
            for i in range(gsr_data.shape[0]):
                processed_data[i] = self._preprocess_single(gsr_data[i])
            return processed_data
        else:
            return self._preprocess_single(gsr_data)
    
    def _preprocess_single(self, gsr_data: np.ndarray) -> np.ndarray:
        """Preprocess a single GSR sample."""
        # Apply bandpass filter
        gsr_data = self._apply_bandpass_filter(gsr_data)
        
        # Smooth the data
        gsr_data = self._smooth_data(gsr_data)
        
        # Normalize
        if self.normalize:
            gsr_data = self._normalize(gsr_data)
        
        return gsr_data
    
    def _apply_bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply bandpass filter for GSR."""
        nyquist = self.sampling_rate / 2
        low = self.low_freq / nyquist
        high = self.high_freq / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered_data[i] = signal.filtfilt(b, a, data[i])
        
        return filtered_data
    
    def _smooth_data(self, data: np.ndarray) -> np.ndarray:
        """Apply smoothing to GSR data."""
        if self.smooth_window > 1:
            smoothed_data = np.zeros_like(data)
            for i in range(data.shape[0]):
                smoothed_data[i] = signal.savgol_filter(
                    data[i], 
                    self.smooth_window, 
                    3 if self.smooth_window >= 3 else 1
                )
            return smoothed_data
        return data
    
    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize GSR data."""
        return zscore(data, axis=1)


class MultiModalPreprocessor:
    """
    Combined preprocessor for multi-modal data.
    """
    
    def __init__(
        self,
        eeg_config: Optional[Dict] = None,
        eye_tracking_config: Optional[Dict] = None,
        gsr_config: Optional[Dict] = None
    ):
        """
        Initialize multi-modal preprocessor.
        
        Args:
            eeg_config: Configuration for EEG preprocessing
            eye_tracking_config: Configuration for eye-tracking preprocessing
            gsr_config: Configuration for GSR preprocessing
        """
        self.eeg_preprocessor = EEGPreprocessor(**(eeg_config or {}))
        self.eye_tracking_preprocessor = EyeTrackingPreprocessor(**(eye_tracking_config or {}))
        self.gsr_preprocessor = GSRPreprocessor(**(gsr_config or {}))
    
    def preprocess(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Preprocess multi-modal data.
        
        Args:
            data: Dictionary containing data for each modality
            
        Returns:
            Dictionary containing preprocessed data for each modality
        """
        processed_data = {}
        
        if 'eeg' in data:
            processed_data['eeg'] = self.eeg_preprocessor.preprocess(data['eeg'])
        
        if 'eye_tracking' in data:
            processed_data['eye_tracking'] = self.eye_tracking_preprocessor.preprocess(data['eye_tracking'])
        
        if 'gsr' in data:
            processed_data['gsr'] = self.gsr_preprocessor.preprocess(data['gsr'])
        
        return processed_data
