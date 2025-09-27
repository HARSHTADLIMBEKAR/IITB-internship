# 🧠 Cross-Modal Knowledge Transfer

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **T1_G15 - GrowAI Project**: Advanced Cross-Modal Knowledge Transfer using EEG, Eye-tracking, and GSR data

## 🎯 Problem Statement

**Objective:** Use EEG to train a model and test if eye-tracking or GSR-only models can approximate it through domain adaptation and modality dropout.

**Advanced Features:** Adversarial domain adaptation and contrastive learning for robust cross-modal transfer.

## ✨ Key Features

- 🔄 **Multi-Modal Data Processing**: Comprehensive support for EEG, eye-tracking, and GSR data
- 🎯 **Domain Adaptation**: Seamless knowledge transfer between different modalities
- 🎲 **Modality Dropout**: Train robust models that work with missing modalities
- ⚔️ **Adversarial Training**: Advanced domain adaptation techniques
- 🔗 **Contrastive Learning**: Learn robust cross-modal representations (SimCLR, MoCo)
- 📊 **Comprehensive Evaluation**: Multiple metrics and visualization tools
- ⚙️ **Flexible Configuration**: YAML-based configuration system
- 🚀 **Production Ready**: Complete training pipeline with checkpointing and monitoring

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   EEG Data      │    │ Eye-tracking    │    │   GSR Data      │
│   (64 channels) │    │   (4 features)  │    │  (1 channel)    │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  EEG Encoder    │    │ Eye Encoder     │    │  GSR Encoder    │
│  (CNN + LSTM)   │    │ (LSTM + Attn)   │    │ (CNN + LSTM)    │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │   Cross-Modal Fusion    │
                    │ (Attention/Concat/Cross)│
                    └─────────┬───────────────┘
                              ▼
                    ┌─────────────────────────┐
                    │   Domain Adaptation     │
                    │  (Linear/MLP/Adversarial)│
                    └─────────┬───────────────┘
                              ▼
                    ┌─────────────────────────┐
                    │    Classification       │
                    │     (2 classes)         │
                    └─────────────────────────┘
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/cross-modal-knowledge-transfer.git
cd cross-modal-knowledge-transfer

# Install dependencies
pip install -r requirements.txt
```

### 2. Basic Usage

```bash
# Train with default configuration (creates dummy data automatically)
python train.py

# Train with cross-modal configuration
python train.py --config configs/cross_modal_config.yaml --experiment_name my_experiment

# Evaluate trained model
python train.py --eval_only --resume results/checkpoints/best_model.pth
```

### 3. Python API

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

## 📁 Project Structure

```
cross-modal-knowledge-transfer/
├── 📁 src/                    # Source code
│   ├── 📁 data/              # Data processing and loading
│   │   ├── dataset.py        # Dataset classes
│   │   ├── preprocessing.py  # Preprocessing pipelines
│   │   └── transforms.py     # Data augmentation
│   ├── 📁 models/            # Model architectures
│   │   ├── base_models.py    # Base encoders
│   │   ├── domain_adaptation.py # Domain adaptation
│   │   ├── contrastive_learning.py # Contrastive learning
│   │   ├── cross_modal.py    # Cross-modal fusion
│   │   └── classifiers.py    # Classification models
│   ├── 📁 training/          # Training pipelines
│   │   ├── trainer.py        # Training classes
│   │   ├── losses.py         # Loss functions
│   │   └── utils.py          # Training utilities
│   ├── 📁 evaluation/        # Evaluation framework
│   │   ├── metrics.py        # Evaluation metrics
│   │   ├── evaluator.py      # Model evaluators
│   │   └── visualization.py  # Visualization tools
│   └── 📁 utils/             # Utility functions
├── 📁 configs/               # Configuration files
│   ├── base_config.yaml      # Basic configuration
│   └── cross_modal_config.yaml # Cross-modal configuration
├── 📁 docs/                  # Documentation
│   ├── dataset_format.md     # Dataset format specification
│   ├── usage_examples.md     # Usage examples
│   └── api_reference.md      # API reference
├── 📁 results/               # Training results (gitignored)
├── 📁 data/                  # Dataset storage (gitignored)
├── 📄 requirements.txt       # Dependencies
├── 📄 train.py              # Main training script
├── 📄 README.md             # This file
├── 📄 QUICK_START.md        # Quick start guide
└── 📄 .gitignore            # Git ignore rules
```

## 📊 Supported Modalities

| Modality | Input Shape | Features | Preprocessing |
|----------|-------------|----------|---------------|
| **EEG** | (64, 1000) | 64 channels, 1000 time points | Bandpass filter, artifact removal, normalization |
| **Eye-tracking** | (4, 1000) | X, Y coordinates, pupil diameter | Blink removal, smoothing, normalization |
| **GSR** | (1, 1000) | 1 channel, 1000 time points | Bandpass filter, smoothing, normalization |

## 🎛️ Configuration

### Basic Configuration

```yaml
# configs/base_config.yaml
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

### Cross-Modal Configuration

```yaml
# configs/cross_modal_config.yaml
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

## 📈 Results

The application provides comprehensive evaluation including:

- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Cross-Modal Metrics**: Modality agreement, feature similarity
- **Domain Adaptation Metrics**: MMD, Wasserstein distance, alignment score
- **Visualization**: Training curves, confusion matrices, feature distributions

## 🔬 Research Applications

This framework supports various research scenarios:

1. **Single Modal Analysis**: Train and evaluate individual modalities
2. **Cross-Modal Transfer**: Transfer knowledge between modalities
3. **Domain Adaptation**: Adapt models across different data domains
4. **Modality Dropout**: Train robust models with missing modalities
5. **Contrastive Learning**: Learn robust cross-modal representations

## 📚 Documentation

- **[Quick Start Guide](QUICK_START.md)**: Get started in minutes
- **[Dataset Format](docs/dataset_format.md)**: Data format specifications
- **[Usage Examples](docs/usage_examples.md)**: Comprehensive examples
- **[API Reference](docs/api_reference.md)**: Complete API documentation

## 🛠️ Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn
- MNE (for EEG processing)

See `requirements.txt` for complete list.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **GrowAI**: For the research opportunity and problem statement
- **PyTorch Team**: For the excellent deep learning framework
- **MNE Team**: For EEG processing capabilities
- **Research Community**: For domain adaptation and contrastive learning techniques

## 📞 Contact

- **Project**: T1_G15 - GrowAI Cross-Modal Knowledge Transfer
- **Repository**: [GitHub Repository](https://github.com/YOUR_USERNAME/cross-modal-knowledge-transfer)
- **Issues**: [GitHub Issues](https://github.com/YOUR_USERNAME/cross-modal-knowledge-transfer/issues)

---

⭐ **Star this repository** if you find it helpful for your research!

🔬 **Perfect for**: Machine Learning researchers, Neuroscience studies, Cross-modal learning experiments
