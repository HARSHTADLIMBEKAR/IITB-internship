# API Reference

This document provides detailed API reference for the Cross-Modal Knowledge Transfer application.

## Data Module (`src.data`)

### Dataset Classes

#### `MultiModalDataset`

```python
class MultiModalDataset(Dataset):
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
    )
```

**Parameters:**
- `data_path`: Path to the dataset directory
- `modalities`: List of available modalities
- `target_modality`: Primary modality for training
- `domain_adaptation`: Whether to use domain adaptation
- `modality_dropout`: Whether to apply modality dropout
- `dropout_prob`: Probability of dropping a modality
- `transform`: Transform to apply to input data
- `target_transform`: Transform to apply to target data

**Methods:**
- `__len__()`: Return the number of samples
- `__getitem__(idx)`: Get a sample from the dataset

#### `EEGDataset`

```python
class EEGDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None
    )
```

#### `EyeTrackingDataset`

```python
class EyeTrackingDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None
    )
```

#### `GSRDataset`

```python
class GSRDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None
    )
```

### Preprocessing Classes

#### `EEGPreprocessor`

```python
class EEGPreprocessor:
    def __init__(
        self,
        sampling_rate: int = 1000,
        low_freq: float = 1.0,
        high_freq: float = 40.0,
        notch_freq: float = 50.0,
        normalize: bool = True,
        remove_artifacts: bool = True
    )
```

**Methods:**
- `preprocess(eeg_data)`: Apply full preprocessing pipeline

#### `EyeTrackingPreprocessor`

```python
class EyeTrackingPreprocessor:
    def __init__(
        self,
        sampling_rate: int = 1000,
        smooth_window: int = 5,
        normalize: bool = True,
        remove_blinks: bool = True
    )
```

#### `GSRPreprocessor`

```python
class GSRPreprocessor:
    def __init__(
        self,
        sampling_rate: int = 1000,
        low_freq: float = 0.05,
        high_freq: float = 2.0,
        normalize: bool = True,
        smooth_window: int = 10
    )
```

### Transform Classes

#### `MultiModalTransform`

```python
class MultiModalTransform:
    def __init__(
        self,
        eeg_transform: Optional[EEGTransform] = None,
        eye_tracking_transform: Optional[EyeTrackingTransform] = None,
        gsr_transform: Optional[GSRTransform] = None,
        modality_dropout: float = 0.0
    )
```

## Models Module (`src.models`)

### Base Models

#### `EEGEncoder`

```python
class EEGEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int = 64,
        sequence_length: int = 1000,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_attention: bool = True
    )
```

**Methods:**
- `forward(x)`: Forward pass through EEG encoder

#### `EyeTrackingEncoder`

```python
class EyeTrackingEncoder(nn.Module):
    def __init__(
        self,
        input_features: int = 4,
        sequence_length: int = 1000,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_attention: bool = True
    )
```

#### `GSREncoder`

```python
class GSREncoder(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        sequence_length: int = 1000,
        hidden_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_attention: bool = True
    )
```

#### `MultiModalEncoder`

```python
class MultiModalEncoder(nn.Module):
    def __init__(
        self,
        eeg_channels: int = 64,
        eye_tracking_features: int = 4,
        gsr_channels: int = 1,
        sequence_length: int = 1000,
        hidden_dim: int = 128,
        fusion_method: str = 'concat',
        dropout: float = 0.1
    )
```

**Parameters:**
- `fusion_method`: Method for fusing modalities ('concat', 'attention', 'cross_attention')

### Domain Adaptation Models

#### `DomainAdapter`

```python
class DomainAdapter(nn.Module):
    def __init__(
        self,
        source_dim: int,
        target_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        adaptation_method: str = 'linear'
    )
```

**Parameters:**
- `adaptation_method`: Method for domain adaptation ('linear', 'mlp', 'residual')

#### `AdversarialDomainAdapter`

```python
class AdversarialDomainAdapter(nn.Module):
    def __init__(
        self,
        source_dim: int,
        target_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        lambda_adv: float = 1.0
    )
```

### Contrastive Learning Models

#### `ContrastiveLearner`

```python
class ContrastiveLearner(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        projection_dim: int = 128,
        temperature: float = 0.07,
        dropout: float = 0.1
    )
```

#### `SimCLR`

```python
class SimCLR(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        projection_dim: int = 128,
        temperature: float = 0.07,
        dropout: float = 0.1
    )
```

#### `MoCo`

```python
class MoCo(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        projection_dim: int = 128,
        queue_size: int = 65536,
        momentum: float = 0.999,
        temperature: float = 0.07,
        dropout: float = 0.1
    )
```

### Classifier Models

#### `MultiModalClassifier`

```python
class MultiModalClassifier(nn.Module):
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
    )
```

#### `CrossModalTransferClassifier`

```python
class CrossModalTransferClassifier(nn.Module):
    def __init__(
        self,
        source_encoder: nn.Module,
        target_encoder: nn.Module,
        num_classes: int = 2,
        hidden_dim: int = 128,
        use_adversarial: bool = False,
        dropout: float = 0.1
    )
```

## Training Module (`src.training`)

### Trainer Classes

#### `Trainer`

```python
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        device: torch.device,
        config: Dict[str, Any]
    )
```

**Methods:**
- `train_epoch()`: Train for one epoch
- `validate_epoch()`: Validate for one epoch
- `train(num_epochs)`: Train the model

#### `CrossModalTrainer`

```python
class CrossModalTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        device: torch.device,
        config: Dict[str, Any]
    )
```

#### `DomainAdaptationTrainer`

```python
class DomainAdaptationTrainer:
    def __init__(
        self,
        model: nn.Module,
        source_dataloader: DataLoader,
        target_dataloader: DataLoader,
        device: torch.device,
        config: Dict[str, Any]
    )
```

### Loss Functions

#### `CrossModalLoss`

```python
class CrossModalLoss(nn.Module):
    def __init__(
        self,
        classification_weight: float = 1.0,
        consistency_weight: float = 0.1,
        alignment_weight: float = 0.1,
        temperature: float = 0.07
    )
```

#### `DomainAdaptationLoss`

```python
class DomainAdaptationLoss(nn.Module):
    def __init__(
        self,
        task_weight: float = 1.0,
        domain_weight: float = 0.1,
        adversarial_weight: float = 0.1
    )
```

#### `ContrastiveLoss`

```python
class ContrastiveLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.07,
        contrastive_weight: float = 1.0,
        classification_weight: float = 1.0
    )
```

### Training Utilities

#### `EarlyStopping`

```python
class EarlyStopping:
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        mode: str = 'min',
        restore_best_weights: bool = True,
        monitor: str = 'val_loss'
    )
```

#### `ModelCheckpoint`

```python
class ModelCheckpoint:
    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        save_weights_only: bool = False,
        verbose: bool = True
    )
```

#### `LearningRateScheduler`

```python
class LearningRateScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        mode: str = 'min',
        factor: float = 0.1,
        patience: int = 10,
        min_lr: float = 1e-7,
        verbose: bool = True
    )
```

## Evaluation Module (`src.evaluation`)

### Evaluator Classes

#### `ModelEvaluator`

```python
class ModelEvaluator:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_classes: int = 2,
        class_names: Optional[List[str]] = None
    )
```

**Methods:**
- `evaluate(dataloader, return_predictions=False)`: Evaluate model on dataset

#### `CrossModalEvaluator`

```python
class CrossModalEvaluator:
    def __init__(
        self,
        models: Dict[str, nn.Module],
        device: torch.device,
        modalities: List[str],
        num_classes: int = 2,
        class_names: Optional[List[str]] = None
    )
```

**Methods:**
- `evaluate_single_modality(modality, dataloader)`: Evaluate single modality model
- `evaluate_cross_modal_transfer(source_modality, target_modality, ...)`: Evaluate cross-modal transfer
- `evaluate_domain_adaptation(source_dataloader, target_dataloader, ...)`: Evaluate domain adaptation

### Metrics Classes

#### `ClassificationMetrics`

```python
class ClassificationMetrics:
    def __init__(
        self,
        num_classes: int = 2,
        class_names: Optional[List[str]] = None
    )
```

**Methods:**
- `update(predictions, targets, probabilities)`: Update metrics with new predictions
- `compute()`: Compute all metrics

#### `CrossModalMetrics`

```python
class CrossModalMetrics:
    def __init__(self, modalities: List[str])
```

#### `DomainAdaptationMetrics`

```python
class DomainAdaptationMetrics:
    def __init__(self)
```

### Visualization Classes

#### `MetricsVisualizer`

```python
class MetricsVisualizer:
    def __init__(self, style: str = 'seaborn-v0_8')
```

**Methods:**
- `plot_metrics_comparison(metrics_dict, metrics_to_plot)`: Plot comparison of metrics
- `plot_training_curves(train_losses, val_losses, ...)`: Plot training curves
- `plot_cross_modal_metrics(cross_modal_results)`: Plot cross-modal metrics

#### `ConfusionMatrixPlotter`

```python
class ConfusionMatrixPlotter:
    def __init__(self, class_names: Optional[List[str]] = None)
```

**Methods:**
- `plot_confusion_matrix(y_true, y_pred, normalize, title)`: Plot confusion matrix
- `plot_multiple_confusion_matrices(confusion_matrices)`: Plot multiple confusion matrices

## Utils Module (`src.utils`)

### Configuration Management

#### `Config`

```python
class Config:
    def __init__(self, config_dict: Dict[str, Any])
```

**Methods:**
- `__getitem__(key)`: Get configuration value
- `__setitem__(key, value)`: Set configuration value
- `get(key, default)`: Get configuration value with default
- `update(other)`: Update configuration with other dictionary
- `save(filepath)`: Save configuration to file
- `load(filepath)`: Load configuration from file

### Device Management

#### Device Functions

```python
def get_device(use_cuda: bool = True, device_id: int = 0) -> torch.device
def set_seed(seed: int = 42)
def get_device_info() -> dict
def print_device_info()
def clear_gpu_memory()
def get_memory_usage() -> dict
def print_memory_usage()
```

## Configuration Files

### Base Configuration (`configs/base_config.yaml`)

```yaml
data:
  data_path: "data/"
  modalities: ["eeg", "eye_tracking", "gsr"]
  target_modality: "eeg"
  batch_size: 32
  num_workers: 4
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15

model:
  type: "multi_modal"
  eeg_channels: 64
  eye_tracking_features: 4
  gsr_channels: 1
  sequence_length: 1000
  hidden_dim: 128
  num_classes: 2
  fusion_method: "attention"
  use_domain_adaptation: false
  modality_dropout: false
  dropout_prob: 0.1

training:
  num_epochs: 100
  learning_rate: 1e-3
  optimizer: "adam"
  weight_decay: 1e-4
  momentum: 0.9
  grad_clip_norm: 1.0
  classification_weight: 1.0
  consistency_weight: 0.1
  alignment_weight: 0.1
  early_stopping_patience: 10
  lr_patience: 5
  monitor_metric: "val_loss"
  checkpoint_path: "results/checkpoints/best_model.pth"
  save_frequency: 10
```

## Command Line Interface

### Training Script (`train.py`)

```bash
python train.py [OPTIONS]

Options:
  --config PATH              Path to configuration file
  --experiment_name NAME     Name of the experiment
  --resume PATH              Path to checkpoint to resume from
  --eval_only               Only evaluate the model
  --help                    Show help message
```

### Example Usage

```bash
# Basic training
python train.py

# Custom configuration
python train.py --config configs/cross_modal_config.yaml --experiment_name my_experiment

# Resume training
python train.py --config configs/base_config.yaml --resume results/checkpoints/best_model.pth

# Evaluation only
python train.py --config configs/base_config.yaml --eval_only --resume results/checkpoints/best_model.pth
```

## Error Handling

### Common Exceptions

#### `ValueError`
- Raised when invalid configuration parameters are provided
- Raised when required data files are missing
- Raised when model architecture parameters are invalid

#### `FileNotFoundError`
- Raised when configuration files cannot be found
- Raised when data files are missing

#### `RuntimeError`
- Raised when CUDA operations fail
- Raised when model training fails

### Error Recovery

```python
try:
    model = create_model(config, device)
except ValueError as e:
    print(f"Configuration error: {e}")
    # Handle configuration error
except RuntimeError as e:
    print(f"Runtime error: {e}")
    # Handle runtime error
```

## Performance Considerations

### Memory Usage

- Use appropriate batch sizes based on available GPU memory
- Enable gradient checkpointing for large models
- Use mixed precision training when available

### Training Speed

- Use multiple workers for data loading
- Enable pin_memory for faster GPU transfer
- Use persistent workers to avoid worker recreation

### Model Size

- Use model pruning for deployment
- Quantize models for inference
- Use knowledge distillation for smaller models
