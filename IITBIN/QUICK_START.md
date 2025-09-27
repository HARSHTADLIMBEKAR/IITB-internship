# Quick Start Guide - Cross-Modal Knowledge Transfer

## 🚀 Getting Started

This application implements Cross-Modal Knowledge Transfer using EEG, eye-tracking, and GSR data for domain adaptation and modality dropout.

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt
```

### 1. Prepare Your Dataset

Place your data in the following structure:
```
data/
├── eeg/
│   └── eeg_data.npy          # Shape: (samples, 64, 1000)
├── eye_tracking/
│   └── eye_data.npy          # Shape: (samples, 4, 1000)
├── gsr/
│   └── gsr_data.npy          # Shape: (samples, 1, 1000)
└── labels.csv                # Columns: label, subject_id
```

**Note**: If you don't have data yet, the application will create dummy datasets automatically for testing.

### 2. Basic Training

```bash
# Train with default configuration
python train.py

# Train with cross-modal configuration
python train.py --config configs/cross_modal_config.yaml --experiment_name cross_modal_experiment
```

### 3. Evaluate Model

```bash
# Evaluate trained model
python train.py --config configs/base_config.yaml --eval_only --resume results/checkpoints/best_model.pth
```

## 🎯 Key Features

### ✅ Implemented Features

1. **Multi-Modal Data Processing**
   - EEG, eye-tracking, and GSR data support
   - Automatic preprocessing pipelines
   - Data augmentation and transforms

2. **Cross-Modal Knowledge Transfer**
   - Domain adaptation between modalities
   - Modality dropout for robust training
   - Adversarial domain adaptation

3. **Advanced Learning Techniques**
   - Contrastive learning (SimCLR, MoCo)
   - Cross-modal attention mechanisms
   - Feature alignment and consistency

4. **Comprehensive Evaluation**
   - Multiple evaluation metrics
   - Cross-modal performance analysis
   - Visualization tools

5. **Flexible Training Pipeline**
   - Configurable training parameters
   - Early stopping and checkpointing
   - Learning rate scheduling

## 📊 Example Results

The application will generate:
- Training curves and metrics
- Confusion matrices
- Cross-modal transfer analysis
- Feature visualizations

## 🔧 Configuration

### Basic Configuration (`configs/base_config.yaml`)
```yaml
model:
  type: "multi_modal"
  fusion_method: "attention"
  use_domain_adaptation: false
  modality_dropout: false

training:
  num_epochs: 100
  learning_rate: 1e-3
  optimizer: "adam"
```

### Cross-Modal Configuration (`configs/cross_modal_config.yaml`)
```yaml
model:
  type: "multi_modal"
  fusion_method: "cross_attention"
  use_domain_adaptation: true
  modality_dropout: true

domain_adaptation:
  enabled: true
  source_modality: "eeg"
  target_modality: "eye_tracking"
```

## 📁 Project Structure

```
├── src/
│   ├── data/              # Data processing and loading
│   ├── models/            # Model architectures
│   ├── training/          # Training pipelines
│   ├── evaluation/        # Evaluation metrics
│   └── utils/             # Utility functions
├── configs/               # Configuration files
├── docs/                  # Documentation
├── results/               # Training results
└── train.py              # Main training script
```

## 🎮 Usage Examples

### Python API
```python
from src.data import MultiModalDataset
from src.models import MultiModalClassifier
from src.training import CrossModalTrainer

# Load data
dataset = MultiModalDataset(
    data_path="data/",
    modalities=['eeg', 'eye_tracking', 'gsr'],
    target_modality='eeg'
)

# Create model
model = MultiModalClassifier(
    eeg_channels=64,
    eye_tracking_features=4,
    gsr_channels=1,
    fusion_method='attention'
)

# Train model
trainer = CrossModalTrainer(model, train_loader, val_loader, device, config)
history = trainer.train(num_epochs=100)
```

### Command Line
```bash
# Single modal training
python train.py --config configs/base_config.yaml

# Cross-modal training with domain adaptation
python train.py --config configs/cross_modal_config.yaml

# Resume training
python train.py --resume results/checkpoints/best_model.pth
```

## 📈 Expected Performance

Based on the implemented techniques:

- **EEG-only**: Baseline performance
- **Eye-tracking-only**: Lower performance than EEG
- **GSR-only**: Lowest performance
- **Cross-modal**: Improved performance through knowledge transfer
- **Domain Adaptation**: Better transfer between modalities

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```yaml
   data:
     batch_size: 16  # Reduce batch size
   ```

2. **Data Not Found**
   - The application will create dummy data automatically
   - Check data format in `docs/dataset_format.md`

3. **Model Not Converging**
   ```yaml
   training:
     learning_rate: 1e-4  # Reduce learning rate
     early_stopping_patience: 15  # Increase patience
   ```

## 📚 Documentation

- **Dataset Format**: `docs/dataset_format.md`
- **Usage Examples**: `docs/usage_examples.md`
- **API Reference**: `docs/api_reference.md`
- **README**: `README.md`

## 🎯 Next Steps

1. **Prepare Your Dataset**: Follow the format in `docs/dataset_format.md`
2. **Run Basic Training**: `python train.py`
3. **Experiment with Configurations**: Try different configs in `configs/`
4. **Analyze Results**: Check `results/` directory for outputs
5. **Customize**: Modify models, losses, or training procedures as needed

## 🆘 Support

If you encounter issues:
1. Check the documentation in `docs/`
2. Verify your data format matches the specification
3. Try with dummy data first: `python train.py`
4. Check the configuration files in `configs/`

---

**Ready to start?** Run `python train.py` to begin training with dummy data!
