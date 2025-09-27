# Dataset Format Specification

This document describes the expected format for multi-modal datasets used in the Cross-Modal Knowledge Transfer application.

## Directory Structure

```
data/
├── eeg/                    # EEG data directory
│   ├── eeg_data.npy       # EEG signals (samples, channels, time_points)
│   └── eeg_info.json      # EEG metadata (optional)
├── eye_tracking/          # Eye-tracking data directory
│   ├── eye_data.npy       # Eye-tracking data (samples, features, time_points)
│   └── eye_info.json      # Eye-tracking metadata (optional)
├── gsr/                   # GSR data directory
│   ├── gsr_data.npy       # GSR signals (samples, channels, time_points)
│   └── gsr_info.json      # GSR metadata (optional)
└── labels.csv             # Labels for all samples
```

## Data Formats

### EEG Data
- **File**: `eeg/eeg_data.npy`
- **Shape**: `(num_samples, num_channels, time_points)`
- **Example**: `(1000, 64, 1000)` for 1000 samples, 64 EEG channels, 1000 time points
- **Data Type**: `float32`
- **Sampling Rate**: Typically 1000 Hz (configurable)

### Eye-Tracking Data
- **File**: `eye_tracking/eye_data.npy`
- **Shape**: `(num_samples, num_features, time_points)`
- **Example**: `(1000, 4, 1000)` for 1000 samples, 4 features (x, y, pupil_l, pupil_r), 1000 time points
- **Features**:
  - `[0]`: X coordinate (normalized)
  - `[1]`: Y coordinate (normalized)
  - `[2]`: Left pupil diameter
  - `[3]`: Right pupil diameter
- **Data Type**: `float32`

### GSR Data
- **File**: `gsr/gsr_data.npy`
- **Shape**: `(num_samples, num_channels, time_points)`
- **Example**: `(1000, 1, 1000)` for 1000 samples, 1 GSR channel, 1000 time points
- **Data Type**: `float32`
- **Sampling Rate**: Typically 1000 Hz (configurable)

### Labels
- **File**: `labels.csv`
- **Format**: CSV with columns:
  - `label`: Class label (integer, 0-based)
  - `subject_id`: Subject identifier (integer)
  - `session_id`: Session identifier (optional, integer)
- **Example**:
  ```csv
  label,subject_id,session_id
  0,1,1
  1,1,1
  0,2,1
  1,2,1
  ```

## Alternative Formats

### CSV Format
If you prefer CSV format, you can use `.csv` files instead of `.npy` files:

```
data/
├── eeg/
│   └── eeg_data.csv       # Shape: (num_samples * time_points, num_channels + 1)
├── eye_tracking/
│   └── eye_data.csv       # Shape: (num_samples * time_points, num_features + 1)
├── gsr/
│   └── gsr_data.csv       # Shape: (num_samples * time_points, num_channels + 1)
└── labels.csv
```

**Note**: CSV files should include a `sample_id` column as the first column to identify which time points belong to which sample.

### NPZ Format
You can also use compressed NumPy format:

```
data/
├── eeg/
│   └── eeg_data.npz       # Contains 'data' key with EEG signals
├── eye_tracking/
│   └── eye_data.npz       # Contains 'data' key with eye-tracking data
├── gsr/
│   └── gsr_data.npz       # Contains 'data' key with GSR data
└── labels.csv
```

## Metadata Files (Optional)

### EEG Info (`eeg/eeg_info.json`)
```json
{
  "sampling_rate": 1000,
  "channels": ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"],
  "units": "microvolts",
  "reference": "average",
  "filtering": {
    "highpass": 0.5,
    "lowpass": 40.0,
    "notch": 50.0
  }
}
```

### Eye-Tracking Info (`eye_tracking/eye_info.json`)
```json
{
  "sampling_rate": 1000,
  "features": ["x_coordinate", "y_coordinate", "left_pupil", "right_pupil"],
  "screen_resolution": [1920, 1080],
  "units": {
    "coordinates": "normalized",
    "pupil": "pixels"
  }
}
```

### GSR Info (`gsr/gsr_info.json`)
```json
{
  "sampling_rate": 1000,
  "channels": ["gsr"],
  "units": "microsiemens",
  "filtering": {
    "highpass": 0.05,
    "lowpass": 2.0
  }
}
```

## Data Preprocessing

The application includes built-in preprocessing for each modality:

### EEG Preprocessing
- DC offset removal
- Notch filtering (50/60 Hz)
- Bandpass filtering (1-40 Hz)
- Artifact removal
- Z-score normalization

### Eye-Tracking Preprocessing
- Blink detection and interpolation
- Smoothing
- Z-score normalization

### GSR Preprocessing
- Bandpass filtering (0.05-2 Hz)
- Smoothing
- Z-score normalization

## Creating Dummy Data

If you don't have real data, the application can create dummy datasets for testing:

```python
from src.data import MultiModalDataset

# This will create dummy data automatically
dataset = MultiModalDataset(
    data_path="dummy_data",  # Non-existent path triggers dummy data creation
    modalities=['eeg', 'eye_tracking', 'gsr'],
    target_modality='eeg'
)
```

## Data Loading Example

```python
import numpy as np
import pandas as pd

# Load EEG data
eeg_data = np.load('data/eeg/eeg_data.npy')
print(f"EEG shape: {eeg_data.shape}")

# Load eye-tracking data
eye_data = np.load('data/eye_tracking/eye_data.npy')
print(f"Eye-tracking shape: {eye_data.shape}")

# Load GSR data
gsr_data = np.load('data/gsr/gsr_data.npy')
print(f"GSR shape: {gsr_data.shape}")

# Load labels
labels = pd.read_csv('data/labels.csv')
print(f"Labels shape: {labels.shape}")
print(labels.head())
```

## Tips for Data Preparation

1. **Synchronization**: Ensure all modalities are synchronized in time
2. **Sampling Rate**: Use consistent sampling rates across modalities
3. **Missing Data**: Handle missing data appropriately (interpolation, exclusion)
4. **Quality Control**: Check for artifacts and noise in the data
5. **Normalization**: Consider the range and distribution of your data
6. **Storage**: Use appropriate data types to save storage space

## Troubleshooting

### Common Issues

1. **Shape Mismatch**: Ensure all modalities have the same number of samples
2. **Missing Files**: Check that all required files exist in the correct locations
3. **Data Type**: Ensure data is in the correct format (float32 for signals)
4. **Labels**: Verify that labels correspond to the correct samples

### Validation

You can validate your dataset format using:

```python
from src.data import MultiModalDataset

try:
    dataset = MultiModalDataset(data_path="your_data_path")
    print(f"Dataset loaded successfully with {len(dataset)} samples")
except Exception as e:
    print(f"Error loading dataset: {e}")
```
