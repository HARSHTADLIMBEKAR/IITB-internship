# Usage Examples

This document provides comprehensive examples of how to use the Cross-Modal Knowledge Transfer application.

## Quick Start

### 1. Basic Training

```bash
# Train with default configuration
python train.py

# Train with custom configuration
python train.py --config configs/cross_modal_config.yaml --experiment_name my_experiment

# Resume training from checkpoint
python train.py --config configs/base_config.yaml --resume results/checkpoints/best_model.pth
```

### 2. Evaluation Only

```bash
# Evaluate trained model
python train.py --config configs/base_config.yaml --eval_only --resume results/checkpoints/best_model.pth
```

## Configuration Examples

### Single Modal Training

```yaml
# configs/eeg_only.yaml
data:
  data_path: "data/"
  modalities: ["eeg"]
  target_modality: "eeg"
  batch_size: 32

model:
  type: "single_modal"
  eeg_channels: 64
  sequence_length: 1000
  hidden_dim: 128
  num_classes: 2

training:
  num_epochs: 50
  learning_rate: 1e-3
  optimizer: "adam"
```

### Cross-Modal Training

```yaml
# configs/cross_modal.yaml
data:
  data_path: "data/"
  modalities: ["eeg", "eye_tracking", "gsr"]
  target_modality: "eeg"
  batch_size: 16

model:
  type: "multi_modal"
  fusion_method: "attention"
  use_domain_adaptation: true
  modality_dropout: true

training:
  classification_weight: 1.0
  consistency_weight: 0.2
  alignment_weight: 0.1
```

### Domain Adaptation

```yaml
# configs/domain_adaptation.yaml
model:
  type: "multi_modal"
  use_domain_adaptation: true

domain_adaptation:
  enabled: true
  source_modality: "eeg"
  target_modality: "eye_tracking"
  adaptation_method: "mlp"
  task_weight: 1.0
  domain_weight: 0.1
  adversarial_weight: 0.1
```

## Python API Examples

### 1. Loading and Preprocessing Data

```python
import torch
from src.data import MultiModalDataset, EEGPreprocessor
from src.data.transforms import MultiModalTransform

# Load dataset
dataset = MultiModalDataset(
    data_path="data/",
    modalities=['eeg', 'eye_tracking', 'gsr'],
    target_modality='eeg',
    domain_adaptation=True,
    modality_dropout=True
)

# Create data loader
dataloader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=32, 
    shuffle=True
)

# Apply preprocessing
eeg_preprocessor = EEGPreprocessor(
    sampling_rate=1000,
    low_freq=1.0,
    high_freq=40.0,
    normalize=True
)

# Apply transforms
transform = MultiModalTransform(
    modality_dropout=0.3
)
```

### 2. Creating Models

```python
from src.models import (
    MultiModalClassifier, EEGClassifier, 
    CrossModalTransferClassifier
)

# Multi-modal classifier
model = MultiModalClassifier(
    eeg_channels=64,
    eye_tracking_features=4,
    gsr_channels=1,
    sequence_length=1000,
    hidden_dim=128,
    num_classes=2,
    fusion_method='attention',
    use_domain_adaptation=True
)

# Single modal classifier
eeg_model = EEGClassifier(
    input_channels=64,
    sequence_length=1000,
    hidden_dim=128,
    num_classes=2
)

# Cross-modal transfer classifier
transfer_model = CrossModalTransferClassifier(
    source_encoder=eeg_model.encoder,
    target_encoder=eye_model.encoder,
    num_classes=2,
    use_adversarial=True
)
```

### 3. Training Models

```python
from src.training import Trainer, CrossModalTrainer
from src.training.losses import CrossModalLoss

# Create trainer
trainer = CrossModalTrainer(
    model=model,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    device=device,
    config=training_config
)

# Train model
history = trainer.train(num_epochs=100)

# Custom loss function
loss_fn = CrossModalLoss(
    classification_weight=1.0,
    consistency_weight=0.1,
    alignment_weight=0.1
)
```

### 4. Evaluation

```python
from src.evaluation import ModelEvaluator, CrossModalEvaluator

# Single model evaluation
evaluator = ModelEvaluator(model, device)
results = evaluator.evaluate(test_loader, return_predictions=True)

print("Accuracy:", results['accuracy'])
print("F1 Score:", results['f1_score'])

# Cross-modal evaluation
cross_modal_evaluator = CrossModalEvaluator(
    models={'eeg': eeg_model, 'eye_tracking': eye_model},
    device=device,
    modalities=['eeg', 'eye_tracking']
)

transfer_results = cross_modal_evaluator.evaluate_cross_modal_transfer(
    source_modality='eeg',
    target_modality='eye_tracking',
    source_dataloader=source_loader,
    target_dataloader=target_loader,
    transfer_model=transfer_model
)
```

### 5. Visualization

```python
from src.evaluation.visualization import MetricsVisualizer, ConfusionMatrixPlotter

# Plot training curves
visualizer = MetricsVisualizer()
visualizer.plot_training_curves(
    train_losses=history['train_loss'],
    val_losses=history['val_loss'],
    train_metrics=history['train_accuracy'],
    val_metrics=history['val_accuracy']
)

# Plot confusion matrix
cm_plotter = ConfusionMatrixPlotter(class_names=['Class 0', 'Class 1'])
cm_plotter.plot_confusion_matrix(
    y_true=results['targets'],
    y_pred=results['predictions']
)

# Plot cross-modal metrics
visualizer.plot_cross_modal_metrics(transfer_results['cross_modal_metrics'])
```

## Advanced Examples

### 1. Custom Loss Function

```python
import torch.nn as nn
from src.training.losses import CrossModalLoss

class CustomCrossModalLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.base_loss = CrossModalLoss()
        self.focal_loss = FocalLoss()
    
    def forward(self, predictions, targets, modality_features):
        # Base cross-modal loss
        base_losses = self.base_loss(predictions, targets, modality_features)
        
        # Custom focal loss for imbalanced classes
        focal_loss = self.focal_loss(predictions['eeg'], targets)
        
        # Combined loss
        total_loss = (
            self.alpha * base_losses['total'] +
            self.beta * focal_loss
        )
        
        return {
            'total': total_loss,
            'base': base_losses['total'],
            'focal': focal_loss
        }
```

### 2. Custom Data Augmentation

```python
from src.data.transforms import TimeSeriesAugmentation

class CustomAugmentation:
    def __init__(self):
        self.ts_aug = TimeSeriesAugmentation(
            jitter_std=0.03,
            scaling_std=0.1,
            time_warping=True
        )
    
    def __call__(self, data, modality):
        if modality == 'eeg':
            # Apply time series augmentation
            return self.ts_aug(data)
        elif modality == 'eye_tracking':
            # Custom eye-tracking augmentation
            return self._augment_eye_tracking(data)
        else:
            return data
    
    def _augment_eye_tracking(self, data):
        # Custom eye-tracking specific augmentation
        # Add gaze noise, simulate blinks, etc.
        return data
```

### 3. Custom Model Architecture

```python
import torch.nn as nn
from src.models.base_models import EEGEncoder

class CustomEEGClassifier(nn.Module):
    def __init__(self, input_channels=64, num_classes=2):
        super().__init__()
        self.encoder = EEGEncoder(input_channels=input_channels)
        
        # Custom attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=self.encoder.output_dim,
            num_heads=8
        )
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.output_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # Encode EEG
        features = self.encoder(x)
        
        # Apply attention
        attended_features, _ = self.attention(
            features.unsqueeze(1), 
            features.unsqueeze(1), 
            features.unsqueeze(1)
        )
        
        # Classify
        logits = self.classifier(attended_features.squeeze(1))
        
        return logits
```

### 4. Hyperparameter Tuning

```python
import itertools
from src.training import Trainer

def hyperparameter_tuning():
    # Define parameter grid
    param_grid = {
        'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3],
        'hidden_dim': [64, 128, 256],
        'batch_size': [16, 32, 64],
        'dropout_prob': [0.1, 0.2, 0.3]
    }
    
    best_score = 0
    best_params = None
    
    # Generate parameter combinations
    param_combinations = list(itertools.product(*param_grid.values()))
    
    for params in param_combinations:
        # Create config with current parameters
        config = {
            'learning_rate': params[0],
            'hidden_dim': params[1],
            'batch_size': params[2],
            'dropout_prob': params[3]
        }
        
        # Train model
        model = create_model(config)
        trainer = Trainer(model, train_loader, val_loader, device, config)
        history = trainer.train(num_epochs=20)  # Short training for tuning
        
        # Evaluate
        evaluator = ModelEvaluator(model, device)
        results = evaluator.evaluate(val_loader)
        score = results['accuracy']
        
        if score > best_score:
            best_score = score
            best_params = config
    
    print(f"Best parameters: {best_params}")
    print(f"Best score: {best_score}")
    
    return best_params
```

### 5. Model Ensemble

```python
class ModelEnsemble(nn.Module):
    def __init__(self, models, weights=None):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.weights = weights or [1.0] * len(models)
    
    def forward(self, x):
        predictions = []
        
        for model in self.models:
            pred = model(x)
            if isinstance(pred, dict):
                pred = pred['logits']
            predictions.append(pred)
        
        # Weighted average
        ensemble_pred = sum(w * p for w, p in zip(self.weights, predictions))
        ensemble_pred /= sum(self.weights)
        
        return ensemble_pred

# Create ensemble
eeg_model = EEGClassifier()
eye_model = EyeTrackingClassifier()
gsr_model = GSRClassifier()

ensemble = ModelEnsemble([eeg_model, eye_model, gsr_model], weights=[0.5, 0.3, 0.2])
```

## Jupyter Notebook Examples

### 1. Data Exploration

```python
# notebooks/data_exploration.ipynb
import matplotlib.pyplot as plt
import numpy as np
from src.data import MultiModalDataset

# Load dataset
dataset = MultiModalDataset(data_path="data/")

# Explore data shapes
sample = dataset[0]
print("Sample keys:", sample.keys())
print("EEG shape:", sample['eeg'].shape)
print("Eye-tracking shape:", sample['eye_tracking'].shape)
print("GSR shape:", sample['gsr'].shape)

# Visualize sample data
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

# EEG
axes[0].plot(sample['eeg'][0, :1000])  # First channel, first 1000 time points
axes[0].set_title('EEG Sample')

# Eye-tracking
axes[1].plot(sample['eye_tracking'][0, :1000])  # First feature
axes[1].set_title('Eye-tracking Sample')

# GSR
axes[2].plot(sample['gsr'][0, 0, :1000])  # First channel
axes[2].set_title('GSR Sample')

plt.tight_layout()
plt.show()
```

### 2. Model Comparison

```python
# notebooks/model_comparison.ipynb
from src.models import EEGClassifier, EyeTrackingClassifier, GSRClassifier
from src.evaluation import ModelEvaluator

# Train different models
models = {
    'EEG': EEGClassifier(),
    'Eye-tracking': EyeTrackingClassifier(),
    'GSR': GSRClassifier()
}

results = {}

for name, model in models.items():
    # Train model (simplified)
    trainer = Trainer(model, train_loader, val_loader, device, config)
    trainer.train(num_epochs=50)
    
    # Evaluate
    evaluator = ModelEvaluator(model, device)
    results[name] = evaluator.evaluate(test_loader)

# Compare results
import pandas as pd

comparison_df = pd.DataFrame(results).T
print(comparison_df[['accuracy', 'f1_score', 'precision', 'recall']])
```

## Command Line Examples

### 1. Batch Training

```bash
#!/bin/bash
# scripts/batch_training.sh

configs=("configs/eeg_only.yaml" "configs/eye_only.yaml" "configs/gsr_only.yaml" "configs/cross_modal.yaml")

for config in "${configs[@]}"; do
    echo "Training with $config"
    python train.py --config "$config" --experiment_name "$(basename $config .yaml)"
done
```

### 2. Hyperparameter Search

```bash
#!/bin/bash
# scripts/hyperparameter_search.sh

learning_rates=(1e-4 5e-4 1e-3 5e-3)
hidden_dims=(64 128 256)

for lr in "${learning_rates[@]}"; do
    for hidden_dim in "${hidden_dims[@]}"; do
        echo "Training with lr=$lr, hidden_dim=$hidden_dim"
        python train.py \
            --config configs/base_config.yaml \
            --experiment_name "lr_${lr}_hidden_${hidden_dim}" \
            --overrides "{\"training\": {\"learning_rate\": $lr}, \"model\": {\"hidden_dim\": $hidden_dim}}"
    done
done
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Reduce batch size
   config['data']['batch_size'] = 16
   
   # Use gradient accumulation
   config['training']['gradient_accumulation_steps'] = 4
   ```

2. **Data Loading Errors**
   ```python
   # Check data format
   dataset = MultiModalDataset(data_path="data/")
   print(f"Dataset size: {len(dataset)}")
   
   # Check sample
   sample = dataset[0]
   print("Sample keys:", sample.keys())
   ```

3. **Model Convergence Issues**
   ```python
   # Adjust learning rate
   config['training']['learning_rate'] = 1e-4
   
   # Use learning rate scheduling
   config['training']['lr_scheduler'] = 'cosine'
   
   # Add regularization
   config['training']['weight_decay'] = 1e-4
   ```

### Performance Tips

1. **Use Mixed Precision Training**
   ```python
   from torch.cuda.amp import GradScaler, autocast
   
   scaler = GradScaler()
   
   with autocast():
       outputs = model(inputs)
       loss = criterion(outputs, targets)
   
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```

2. **Optimize Data Loading**
   ```python
   dataloader = DataLoader(
       dataset,
       batch_size=32,
       num_workers=4,
       pin_memory=True,
       persistent_workers=True
   )
   ```

3. **Use Model Checkpointing**
   ```python
   # Save best model
   if val_accuracy > best_accuracy:
       torch.save(model.state_dict(), 'best_model.pth')
       best_accuracy = val_accuracy
   ```
