"""
Evaluation Metrics for Mirror Descent Implicit Bias Analysis.

This module implements various metrics to analyze the implicit bias properties
of models trained with Mirror Descent optimization.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch.nn.functional as F


def margin(u: torch.Tensor, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute margin: min_i y_i(u^T x_i) / ||u||.
    
    Args:
        u: Weight vector
        X: Input features
        y: Target labels (-1 or +1)
        
    Returns:
        Margin value
    """
    # Compute predictions
    predictions = X @ u
    
    # Compute margins: y_i * (u^T x_i)
    margins = y * predictions
    
    # Return minimum margin
    return torch.min(margins)


def angle(u: torch.Tensor, v: torch.Tensor) -> float:
    """
    Compute angle between two vectors in degrees, with safe clamping.
    
    Args:
        u: First vector
        v: Second vector
        
    Returns:
        Angle in degrees
    """
    # Compute cosine similarity
    cos_sim = torch.dot(u, v) / (torch.norm(u) * torch.norm(v))
    
    # Clamp to avoid numerical issues
    cos_sim = torch.clamp(cos_sim, min=-1.0, max=1.0)
    
    # Convert to degrees
    angle_rad = torch.acos(cos_sim)
    angle_deg = angle_rad * 180.0 / torch.pi
    
    return angle_deg.item()


def angles_to_baselines(u: torch.Tensor, 
                       X: torch.Tensor, 
                       y: torch.Tensor,
                       baselines: Dict[str, torch.Tensor],
                       linear_md: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """
    Compute angles to baseline methods (SVM, logistic regression, and linear-MD if provided).
    
    Args:
        u: Deep network effective weight
        X: Input features
        y: Target labels
        baselines: Dictionary of baseline weight vectors
        linear_md: Optional linear Mirror Descent weight
        
    Returns:
        Dictionary of angles to baselines
    """
    angles = {}
    
    for name, baseline_w in baselines.items():
        angles[f'angle_to_{name}'] = angle(u, baseline_w)
    
    if linear_md is not None:
        angles['angle_to_linear_md'] = angle(u, linear_md)
    
    return angles


def layer_alignment(W_list: List[torch.Tensor]) -> float:
    """
    Compute cosine similarity between top singular vectors across layers.
    
    Args:
        W_list: List of weight matrices for each layer
        
    Returns:
        Average alignment across layers
    """
    if len(W_list) < 2:
        return 1.0
    
    alignments = []
    
    for i in range(len(W_list) - 1):
        W1, W2 = W_list[i], W_list[i + 1]
        
        # Compute SVD
        U1, _, _ = torch.svd(W1)
        U2, _, _ = torch.svd(W2)
        
        # Get top singular vector
        u1 = U1[:, 0]
        u2 = U2[:, 0]
        
        # Compute cosine similarity
        cos_sim = torch.dot(u1, u2) / (torch.norm(u1) * torch.norm(u2))
        cos_sim = torch.clamp(cos_sim, min=-1.0, max=1.0)
        
        alignments.append(cos_sim.item())
    
    return np.mean(alignments)


def norm_balance(W_list: List[torch.Tensor]) -> float:
    """
    Compute max/min Frobenius norms across layers.
    
    Args:
        W_list: List of weight matrices for each layer
        
    Returns:
        Norm balance ratio (max/min)
    """
    norms = [torch.norm(W, p='fro').item() for W in W_list]
    
    if min(norms) == 0:
        return float('inf')
    
    return max(norms) / min(norms)


def ntk_drift(model: torch.nn.Module, 
              X: torch.Tensor,
              steps: int = 100) -> Dict[str, float]:
    """
    Compute Neural Tangent Kernel drift during training.
    
    TODO: Implement NTK drift computation with docstring.
    This would track how the NTK evolves during training to understand
    the implicit bias dynamics.
    
    Args:
        model: Model to analyze
        X: Input features
        steps: Number of steps to track
        
    Returns:
        Dictionary of NTK drift metrics
    """
    # TODO: Implement NTK drift computation
    # This is a placeholder implementation
    return {
        'ntk_drift_mean': 0.0,
        'ntk_drift_std': 0.0,
        'note': 'NTK drift computation not yet implemented'
    }


def compute_spectral_properties(model: torch.nn.Module) -> Dict[str, float]:
    """
    Compute spectral properties of the model weights.
    
    Args:
        model: Trained model
        
    Returns:
        Dictionary of spectral properties
    """
    properties = {}
    
    if hasattr(model, 'get_layer_weights'):
        W_list = model.get_layer_weights()
        
        for i, W in enumerate(W_list):
            # Singular values
            _, S, _ = torch.svd(W)
            properties[f'layer_{i}_max_singular'] = S[0].item()
            properties[f'layer_{i}_min_singular'] = S[-1].item()
            properties[f'layer_{i}_condition_number'] = (S[0] / S[-1]).item()
            
            # Frobenius norm
            properties[f'layer_{i}_frobenius_norm'] = torch.norm(W, p='fro').item()
    
    return properties


def compute_generalization_gap(train_features: torch.Tensor,
                             train_labels: torch.Tensor,
                             test_features: torch.Tensor,
                             test_labels: torch.Tensor,
                             model: torch.nn.Module) -> float:
    """
    Compute the generalization gap between training and test performance.
    
    Args:
        train_features: Training features
        train_labels: Training labels
        test_features: Test features
        test_labels: Test labels
        model: Trained model
        
    Returns:
        Generalization gap
    """
    model.eval()
    
    with torch.no_grad():
        # Training accuracy
        train_pred = model(train_features)
        train_acc = torch.mean((torch.sign(train_pred) == train_labels).float()).item()
        
        # Test accuracy
        test_pred = model(test_features)
        test_acc = torch.mean((torch.sign(test_pred) == test_labels).float()).item()
    
    return train_acc - test_acc


def compute_robustness_metrics(features: torch.Tensor,
                             labels: torch.Tensor,
                             model: torch.nn.Module,
                             epsilon: float = 0.1) -> Dict[str, float]:
    """
    Compute robustness metrics for the model.
    
    Args:
        features: Input features
        labels: Target labels
        model: Trained model
        epsilon: Perturbation magnitude
        
    Returns:
        Dictionary of robustness metrics
    """
    model.eval()
    
    with torch.no_grad():
        # Original predictions
        original_pred = model(features)
        original_acc = torch.mean((torch.sign(original_pred) == labels).float()).item()
        
        # Add random perturbations
        noise = torch.randn_like(features) * epsilon
        perturbed_features = features + noise
        
        # Perturbed predictions
        perturbed_pred = model(perturbed_features)
        perturbed_acc = torch.mean((torch.sign(perturbed_pred) == labels).float()).item()
        
        # Robustness gap
        robustness_gap = original_acc - perturbed_acc
    
    return {
        'original_accuracy': original_acc,
        'perturbed_accuracy': perturbed_acc,
        'robustness_gap': robustness_gap
    }


def analyze_implicit_bias(features: torch.Tensor,
                        labels: torch.Tensor,
                        model: torch.nn.Module,
                        potential_name: str) -> Dict[str, float]:
    """
    Comprehensive analysis of implicit bias properties.
    
    Args:
        features: Input features
        labels: Target labels
        model: Trained model
        potential_name: Name of the potential function used
        
    Returns:
        Dictionary of implicit bias metrics
    """
    metrics = {}
    
    # Get effective weight if available
    if hasattr(model, 'effective_weight'):
        u = model.effective_weight()
        
        # Margin
        metrics['margin'] = margin(u, features, labels).item()
        
        # Spectral properties
        spectral_props = compute_spectral_properties(model)
        metrics.update(spectral_props)
        
        # Layer alignment if deep network
        if hasattr(model, 'get_layer_weights'):
            W_list = model.get_layer_weights()
            metrics['layer_alignment'] = layer_alignment(W_list)
            metrics['norm_balance'] = norm_balance(W_list)
    
    # General properties
    metrics['potential_name'] = potential_name
    metrics['model_type'] = type(model).__name__
    
    return metrics


class MetricsCollector:
    """
    Utility class for collecting and aggregating evaluation metrics.
    """
    
    def __init__(self):
        """Initialize metrics collector."""
        self.metrics_history = {}
        
    def add_metrics(self, metrics: Dict[str, float], experiment_id: str):
        """
        Add metrics for an experiment.
        
        Args:
            metrics: Dictionary of metrics
            experiment_id: Unique identifier for the experiment
        """
        self.metrics_history[experiment_id] = metrics
        
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics across all experiments.
        
        Returns:
            Dictionary of summary statistics
        """
        if not self.metrics_history:
            return {}
        
        # Collect all metric names
        all_metrics = set()
        for metrics in self.metrics_history.values():
            all_metrics.update(metrics.keys())
        
        summary = {}
        for metric_name in all_metrics:
            values = []
            for metrics in self.metrics_history.values():
                if metric_name in metrics and isinstance(metrics[metric_name], (int, float)):
                    values.append(metrics[metric_name])
            
            if values:
                summary[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
        
        return summary
        
    def save_metrics(self, filepath: str):
        """
        Save collected metrics to file.
        
        Args:
            filepath: Path to save metrics
        """
        import json
        
        # Convert to serializable format
        serializable_data = {
            'metrics_history': self.metrics_history,
            'summary': self.get_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(serializable_data, f, indent=2)
