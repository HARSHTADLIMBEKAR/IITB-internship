"""
Visualization utilities for evaluation results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class MetricsVisualizer:
    """
    Visualization utilities for evaluation metrics.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        """
        Initialize metrics visualizer.
        
        Args:
            style: Matplotlib style
        """
        plt.style.use(style)
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    def plot_metrics_comparison(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        metrics_to_plot: List[str] = ['accuracy', 'precision', 'recall', 'f1_score'],
        save_path: Optional[str] = None
    ):
        """
        Plot comparison of metrics across different models/modalities.
        
        Args:
            metrics_dict: Dictionary mapping model names to metrics
            metrics_to_plot: List of metrics to plot
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]
            
            models = list(metrics_dict.keys())
            values = [metrics_dict[model].get(metric, 0) for model in models]
            
            bars = ax.bar(models, values, color=self.colors[:len(models)])
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
            
            # Rotate x-axis labels if needed
            if len(max(models, key=len)) > 10:
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_training_curves(
        self,
        train_losses: List[float],
        val_losses: List[float],
        train_metrics: Optional[List[float]] = None,
        val_metrics: Optional[List[float]] = None,
        metric_name: str = 'Accuracy',
        save_path: Optional[str] = None
    ):
        """
        Plot training and validation curves.
        
        Args:
            train_losses: Training losses
            val_losses: Validation losses
            train_metrics: Training metrics
            val_metrics: Validation metrics
            metric_name: Name of the metric
            save_path: Path to save the plot
        """
        epochs = range(1, len(train_losses) + 1)
        
        if train_metrics is not None and val_metrics is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Loss plot
            ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
            ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Metrics plot
            ax2.plot(epochs, train_metrics, 'b-', label=f'Training {metric_name}')
            ax2.plot(epochs, val_metrics, 'r-', label=f'Validation {metric_name}')
            ax2.set_title(f'Training and Validation {metric_name}')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel(metric_name)
            ax2.legend()
            ax2.grid(True)
        else:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            ax.plot(epochs, train_losses, 'b-', label='Training Loss')
            ax.plot(epochs, val_losses, 'r-', label='Validation Loss')
            ax.set_title('Training and Validation Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_cross_modal_metrics(
        self,
        cross_modal_results: Dict[str, Any],
        save_path: Optional[str] = None
    ):
        """
        Plot cross-modal evaluation metrics.
        
        Args:
            cross_modal_results: Cross-modal evaluation results
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Modality performance
        if 'modality_performance' in cross_modal_results:
            ax = axes[0]
            modality_perf = cross_modal_results['modality_performance']
            
            modalities = list(modality_perf.keys())
            accuracies = [modality_perf[mod]['accuracy'] for mod in modalities]
            
            bars = ax.bar(modalities, accuracies, color=self.colors[:len(modalities)])
            ax.set_title('Modality Performance (Accuracy)')
            ax.set_ylabel('Accuracy')
            ax.set_ylim(0, 1)
            
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{acc:.3f}', ha='center', va='bottom')
        
        # Cross-modal consistency
        if 'consistency' in cross_modal_results:
            ax = axes[1]
            consistency = cross_modal_results['consistency']
            
            pairs = list(consistency.keys())
            agreements = list(consistency.values())
            
            bars = ax.bar(pairs, agreements, color=self.colors[:len(pairs)])
            ax.set_title('Cross-Modal Agreement')
            ax.set_ylabel('Agreement Score')
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)
            
            for bar, agreement in zip(bars, agreements):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{agreement:.3f}', ha='center', va='bottom')
        
        # Feature similarity
        if 'feature_similarity' in cross_modal_results:
            ax = axes[2]
            similarity = cross_modal_results['feature_similarity']
            
            pairs = list(similarity.keys())
            similarities = list(similarity.values())
            
            bars = ax.bar(pairs, similarities, color=self.colors[:len(pairs)])
            ax.set_title('Feature Similarity')
            ax.set_ylabel('Cosine Similarity')
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)
            
            for bar, sim in zip(bars, similarities):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{sim:.3f}', ha='center', va='bottom')
        
        # Hide unused subplot
        axes[3].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_domain_adaptation_metrics(
        self,
        domain_metrics: Dict[str, float],
        save_path: Optional[str] = None
    ):
        """
        Plot domain adaptation metrics.
        
        Args:
            domain_metrics: Domain adaptation metrics
            save_path: Path to save the plot
        """
        metrics = list(domain_metrics.keys())
        values = list(domain_metrics.values())
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        bars = ax.bar(metrics, values, color=self.colors[:len(metrics)])
        ax.set_title('Domain Adaptation Metrics')
        ax.set_ylabel('Score')
        ax.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class ConfusionMatrixPlotter:
    """
    Confusion matrix plotting utilities.
    """
    
    def __init__(self, class_names: Optional[List[str]] = None):
        """
        Initialize confusion matrix plotter.
        
        Args:
            class_names: Names of classes
        """
        self.class_names = class_names
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        normalize: bool = False,
        title: str = 'Confusion Matrix',
        save_path: Optional[str] = None
    ):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Whether to normalize the matrix
            title: Plot title
            save_path: Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_multiple_confusion_matrices(
        self,
        confusion_matrices: Dict[str, np.ndarray],
        normalize: bool = False,
        save_path: Optional[str] = None
    ):
        """
        Plot multiple confusion matrices.
        
        Args:
            confusion_matrices: Dictionary mapping names to confusion matrices
            normalize: Whether to normalize the matrices
            save_path: Path to save the plot
        """
        n_matrices = len(confusion_matrices)
        cols = min(3, n_matrices)
        rows = (n_matrices + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, (name, cm) in enumerate(confusion_matrices.items()):
            ax = axes[i]
            
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                fmt = '.2f'
            else:
                fmt = 'd'
            
            sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                       xticklabels=self.class_names,
                       yticklabels=self.class_names,
                       ax=ax)
            
            ax.set_title(name)
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
        
        # Hide unused subplots
        for i in range(n_matrices, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class FeatureVisualizer:
    """
    Feature visualization utilities.
    """
    
    def __init__(self):
        """Initialize feature visualizer."""
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    def plot_feature_distribution(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: Optional[List[str]] = None,
        method: str = 'tsne',  # 'tsne' or 'pca'
        save_path: Optional[str] = None
    ):
        """
        Plot feature distribution using dimensionality reduction.
        
        Args:
            features: Feature matrix
            labels: Class labels
            feature_names: Names of features
            method: Dimensionality reduction method
            save_path: Path to save the plot
        """
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
            reduced_features = reducer.fit_transform(features)
        elif method == 'pca':
            reducer = PCA(n_components=2)
            reduced_features = reducer.fit_transform(features)
        else:
            raise ValueError("Method must be 'tsne' or 'pca'")
        
        unique_labels = np.unique(labels)
        
        plt.figure(figsize=(10, 8))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(reduced_features[mask, 0], reduced_features[mask, 1],
                       c=self.colors[i % len(self.colors)], label=f'Class {label}',
                       alpha=0.7, s=50)
        
        plt.title(f'Feature Distribution ({method.upper()})')
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_cross_modal_features(
        self,
        modality_features: Dict[str, np.ndarray],
        labels: np.ndarray,
        method: str = 'tsne',
        save_path: Optional[str] = None
    ):
        """
        Plot features from different modalities.
        
        Args:
            modality_features: Dictionary mapping modality names to features
            labels: Class labels
            method: Dimensionality reduction method
            save_path: Path to save the plot
        """
        n_modalities = len(modality_features)
        fig, axes = plt.subplots(1, n_modalities, figsize=(5*n_modalities, 5))
        
        if n_modalities == 1:
            axes = [axes]
        
        for i, (modality, features) in enumerate(modality_features.items()):
            ax = axes[i]
            
            if method == 'tsne':
                reducer = TSNE(n_components=2, random_state=42)
                reduced_features = reducer.fit_transform(features)
            elif method == 'pca':
                reducer = PCA(n_components=2)
                reduced_features = reducer.fit_transform(features)
            
            unique_labels = np.unique(labels)
            
            for j, label in enumerate(unique_labels):
                mask = labels == label
                ax.scatter(reduced_features[mask, 0], reduced_features[mask, 1],
                          c=self.colors[j % len(self.colors)], label=f'Class {label}',
                          alpha=0.7, s=50)
            
            ax.set_title(f'{modality.title()} Features ({method.upper()})')
            ax.set_xlabel(f'{method.upper()} Component 1')
            ax.set_ylabel(f'{method.upper()} Component 2')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_interactive_features(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        method: str = 'tsne',
        save_path: Optional[str] = None
    ):
        """
        Create interactive feature plot using Plotly.
        
        Args:
            features: Feature matrix
            labels: Class labels
            method: Dimensionality reduction method
            save_path: Path to save the plot
        """
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
            reduced_features = reducer.fit_transform(features)
        elif method == 'pca':
            reducer = PCA(n_components=2)
            reduced_features = reducer.fit_transform(features)
        
        # Create DataFrame for Plotly
        df = pd.DataFrame({
            'x': reduced_features[:, 0],
            'y': reduced_features[:, 1],
            'label': labels
        })
        
        fig = px.scatter(df, x='x', y='y', color='label',
                        title=f'Interactive Feature Distribution ({method.upper()})',
                        labels={'x': f'{method.upper()} Component 1',
                               'y': f'{method.upper()} Component 2'})
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
