#!/usr/bin/env python3
"""
Main training script for cross-modal knowledge transfer.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils import Config, load_config, get_device, set_seed
from src.data import MultiModalDataset, EEGDataset, EyeTrackingDataset, GSRDataset
from src.models import (
    MultiModalClassifier, EEGClassifier, EyeTrackingClassifier, GSRClassifier,
    CrossModalTransferClassifier
)
from src.training import Trainer, CrossModalTrainer, DomainAdaptationTrainer
from src.evaluation import ModelEvaluator, CrossModalEvaluator
from src.utils.device import print_device_info, print_memory_usage


def create_model(config: Config, device: torch.device) -> nn.Module:
    """
    Create model based on configuration.
    
    Args:
        config: Configuration object
        device: Device to create model on
        
    Returns:
        Created model
    """
    model_config = config['model']
    
    if model_config['type'] == 'multi_modal':
        model = MultiModalClassifier(
            eeg_channels=model_config['eeg_channels'],
            eye_tracking_features=model_config['eye_tracking_features'],
            gsr_channels=model_config['gsr_channels'],
            sequence_length=model_config['sequence_length'],
            hidden_dim=model_config['hidden_dim'],
            num_classes=model_config['num_classes'],
            fusion_method=model_config['fusion_method'],
            use_domain_adaptation=model_config['use_domain_adaptation'],
            modality_dropout=model_config['modality_dropout'],
            dropout_prob=model_config['dropout_prob']
        )
    elif model_config['type'] == 'single_modal':
        # Create single modal model based on target modality
        target_modality = config['data']['target_modality']
        
        if target_modality == 'eeg':
            model = EEGClassifier(
                input_channels=model_config['eeg_channels'],
                sequence_length=model_config['sequence_length'],
                hidden_dim=model_config['hidden_dim'],
                num_classes=model_config['num_classes'],
                dropout=model_config['dropout_prob']
            )
        elif target_modality == 'eye_tracking':
            model = EyeTrackingClassifier(
                input_features=model_config['eye_tracking_features'],
                sequence_length=model_config['sequence_length'],
                hidden_dim=model_config['hidden_dim'],
                num_classes=model_config['num_classes'],
                dropout=model_config['dropout_prob']
            )
        elif target_modality == 'gsr':
            model = GSRClassifier(
                input_channels=model_config['gsr_channels'],
                sequence_length=model_config['sequence_length'],
                hidden_dim=model_config['hidden_dim'],
                num_classes=model_config['num_classes'],
                dropout=model_config['dropout_prob']
            )
        else:
            raise ValueError(f"Unknown target modality: {target_modality}")
    else:
        raise ValueError(f"Unknown model type: {model_config['type']}")
    
    return model.to(device)


def create_datasets(config: Config):
    """
    Create datasets based on configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    data_config = config['data']
    data_path = data_config['data_path']
    
    # Check if data exists
    if not os.path.exists(data_path):
        print(f"Warning: Data path {data_path} does not exist. Creating dummy datasets.")
        return create_dummy_datasets(config)
    
    if config['model']['type'] == 'multi_modal':
        # Create multi-modal dataset
        train_dataset = MultiModalDataset(
            data_path=data_path,
            modalities=data_config['modalities'],
            target_modality=data_config['target_modality'],
            domain_adaptation=config.get('domain_adaptation', {}).get('enabled', False),
            modality_dropout=config['model']['modality_dropout'],
            dropout_prob=0.3
        )
        
        # Split dataset
        total_size = len(train_dataset)
        train_size = int(data_config['train_split'] * total_size)
        val_size = int(data_config['val_split'] * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size, test_size]
        )
        
    else:
        # Create single modal dataset
        target_modality = data_config['target_modality']
        
        if target_modality == 'eeg':
            train_dataset = EEGDataset(data_path=data_path)
        elif target_modality == 'eye_tracking':
            train_dataset = EyeTrackingDataset(data_path=data_path)
        elif target_modality == 'gsr':
            train_dataset = GSRDataset(data_path=data_path)
        else:
            raise ValueError(f"Unknown target modality: {target_modality}")
        
        # Split dataset
        total_size = len(train_dataset)
        train_size = int(data_config['train_split'] * total_size)
        val_size = int(data_config['val_split'] * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size, test_size]
        )
    
    return train_dataset, val_dataset, test_dataset


def create_dummy_datasets(config: Config):
    """
    Create dummy datasets for testing.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    print("Creating dummy datasets for testing...")
    
    data_config = config['data']
    
    if config['model']['type'] == 'multi_modal':
        # Create dummy multi-modal dataset
        train_dataset = MultiModalDataset(
            data_path="dummy_data",
            modalities=data_config['modalities'],
            target_modality=data_config['target_modality'],
            domain_adaptation=config.get('domain_adaptation', {}).get('enabled', False),
            modality_dropout=config['model']['modality_dropout'],
            dropout_prob=0.3
        )
        
        # Split dataset
        total_size = len(train_dataset)
        train_size = int(data_config['train_split'] * total_size)
        val_size = int(data_config['val_split'] * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size, test_size]
        )
        
    else:
        # Create dummy single modal dataset
        target_modality = data_config['target_modality']
        
        if target_modality == 'eeg':
            train_dataset = EEGDataset(data_path="dummy_data")
        elif target_modality == 'eye_tracking':
            train_dataset = EyeTrackingDataset(data_path="dummy_data")
        elif target_modality == 'gsr':
            train_dataset = GSRDataset(data_path="dummy_data")
        else:
            raise ValueError(f"Unknown target modality: {target_modality}")
        
        # Split dataset
        total_size = len(train_dataset)
        train_size = int(data_config['train_split'] * total_size)
        val_size = int(data_config['val_split'] * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size, test_size]
        )
    
    return train_dataset, val_dataset, test_dataset


def create_data_loaders(config: Config, train_dataset, val_dataset, test_dataset):
    """
    Create data loaders.
    
    Args:
        config: Configuration object
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_config = config['data']
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config['batch_size'],
        shuffle=True,
        num_workers=data_config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train cross-modal knowledge transfer model')
    parser.add_argument('--config', type=str, default='configs/base_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--experiment_name', type=str, default='cross_modal_experiment',
                       help='Name of the experiment')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--eval_only', action='store_true',
                       help='Only evaluate the model')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed
    set_seed(config.get('seed', 42))
    
    # Get device
    device_config = config['device']
    device = get_device(
        use_cuda=device_config['use_cuda'],
        device_id=device_config['device_id']
    )
    
    # Print device information
    print_device_info()
    print_memory_usage()
    
    # Create datasets
    print("Creating datasets...")
    train_dataset, val_dataset, test_dataset = create_datasets(config)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        config, train_dataset, val_dataset, test_dataset
    )
    
    # Create model
    print("Creating model...")
    model = create_model(config, device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    if args.eval_only:
        # Load checkpoint
        if args.resume:
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from {args.resume}")
        
        # Evaluate model
        print("Evaluating model...")
        evaluator = ModelEvaluator(model, device)
        results = evaluator.evaluate(test_loader, return_predictions=True)
        
        print("Evaluation Results:")
        for metric, value in results.items():
            if metric not in ['predictions', 'targets', 'probabilities']:
                print(f"{metric}: {value:.4f}")
        
        return
    
    # Create trainer
    print("Creating trainer...")
    if config['model']['type'] == 'multi_modal':
        trainer = CrossModalTrainer(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            device=device,
            config=config['training']
        )
    else:
        trainer = Trainer(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            device=device,
            config=config['training']
        )
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Resumed training from {args.resume}")
    
    # Train model
    print("Starting training...")
    history = trainer.train(config['training']['num_epochs'])
    
    # Save training history
    history_path = f"results/history_{args.experiment_name}.json"
    import json
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")
    
    # Evaluate final model
    print("Evaluating final model...")
    evaluator = ModelEvaluator(model, device)
    results = evaluator.evaluate(test_loader, return_predictions=True)
    
    print("Final Evaluation Results:")
    for metric, value in results.items():
        if metric not in ['predictions', 'targets', 'probabilities']:
            print(f"{metric}: {value:.4f}")
    
    # Save evaluation results
    results_path = f"results/evaluation_{args.experiment_name}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Evaluation results saved to {results_path}")
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
