"""
Synthetic Data Generation for Mirror Descent Experiments.

This module provides utilities for generating synthetic datasets that are
well-suited for studying the implicit bias of Mirror Descent in deep linear networks.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict, Any


def generate_gaussian_data(n: int,
                          d: int,
                          separation: float = 1.0,
                          noise: float = 0.1,
                          seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate two Gaussians with means ±μ, ‖μ‖=separation, covariance σ²I, labels ±1.
    
    Args:
        n: Number of samples to generate
        d: Input feature dimension
        separation: Distance between class centers (‖μ‖)
        noise: Standard deviation of noise (σ)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (features, labels) with labels ±1
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Generate class centers: ±μ where ‖μ‖ = separation
    # Use first dimension for separation, others zero
    mu = torch.zeros(d)
    mu[0] = separation / 2.0  # Half separation in each direction
    
    # Generate samples for each class
    n_per_class = n // 2
    remaining = n - 2 * n_per_class
    
    # Positive class (label +1)
    X_pos = torch.randn(n_per_class + remaining, d) * noise
    X_pos[:, 0] += mu[0]
    y_pos = torch.ones(n_per_class + remaining)
    
    # Negative class (label -1)
    X_neg = torch.randn(n_per_class, d) * noise
    X_neg[:, 0] -= mu[0]
    y_neg = -torch.ones(n_per_class)
    
    # Combine and shuffle
    X = torch.cat([X_pos, X_neg], dim=0)
    y = torch.cat([y_pos, y_neg], dim=0)
    
    # Shuffle
    perm = torch.randperm(n)
    X = X[perm]
    y = y[perm]
    
    return X, y


def standardize(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Standardize features to zero mean and unit variance.
    
    Args:
        X: Input features
        
    Returns:
        Tuple of (standardized_features, mean, std)
    """
    mean = torch.mean(X, dim=0)
    std = torch.std(X, dim=0)
    std = torch.clamp(std, min=1e-8)  # Avoid division by zero
    X_std = (X - mean) / std
    return X_std, mean, std


def train_val_split(X: torch.Tensor, 
                   y: torch.Tensor,
                   val_ratio: float = 0.2,
                   seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split data into train and validation sets.
    
    Args:
        X: Input features
        y: Target labels
        val_ratio: Fraction of data to use for validation
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val)
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    n = X.shape[0]
    n_val = int(n * val_ratio)
    n_train = n - n_val
    
    # Shuffle indices
    perm = torch.randperm(n)
    
    # Split
    train_indices = perm[:n_train]
    val_indices = perm[n_train:]
    
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_val = X[val_indices]
    y_val = y[val_indices]
    
    return X_train, y_train, X_val, y_val


def generate_imbalanced_data(n_samples: int,
                           input_dim: int,
                           imbalance_ratio: float = 0.1,
                           seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate imbalanced Gaussian data.
    
    Args:
        n_samples: Total number of samples
        input_dim: Input feature dimension
        imbalance_ratio: Ratio of minority to majority class
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (features, labels)
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Calculate class sizes
    n_minority = int(n_samples * imbalance_ratio / (1 + imbalance_ratio))
    n_majority = n_samples - n_minority
    
    # Generate minority class (label +1)
    X_minority = torch.randn(n_minority, input_dim) * 0.5
    X_minority[:, 0] += 1.0
    y_minority = torch.ones(n_minority)
    
    # Generate majority class (label -1)
    X_majority = torch.randn(n_majority, input_dim) * 0.5
    X_majority[:, 0] -= 1.0
    y_majority = -torch.ones(n_majority)
    
    # Combine and shuffle
    X = torch.cat([X_minority, X_majority], dim=0)
    y = torch.cat([y_minority, y_majority], dim=0)
    
    perm = torch.randperm(n_samples)
    X = X[perm]
    y = y[perm]
    
    return X, y


def generate_correlated_data(n_samples: int,
                           input_dim: int,
                           correlation_strength: float = 0.5,
                           seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate data with correlated features.
    
    Args:
        n_samples: Number of samples to generate
        input_dim: Input feature dimension
        correlation_strength: Strength of feature correlation
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (features, labels)
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Create correlation matrix
    corr_matrix = torch.eye(input_dim)
    for i in range(input_dim - 1):
        corr_matrix[i, i+1] = correlation_strength
        corr_matrix[i+1, i] = correlation_strength
    
    # Generate correlated noise
    L = torch.linalg.cholesky(corr_matrix)
    noise = torch.randn(n_samples, input_dim) @ L.T
    
    # Add class separation
    labels = 2 * torch.randint(0, 2, (n_samples,)) - 1
    X = noise + labels.unsqueeze(1) * 0.5
    
    return X, labels


def generate_multiclass_data(n_samples: int,
                           input_dim: int,
                           n_classes: int = 3,
                           seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate multiclass Gaussian data.
    
    Args:
        n_samples: Number of samples to generate
        input_dim: Input feature dimension
        n_classes: Number of classes
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (features, labels)
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    n_per_class = n_samples // n_classes
    remaining = n_samples - n_classes * n_per_class
    
    X_list = []
    y_list = []
    
    for c in range(n_classes):
        n_class = n_per_class + (1 if c < remaining else 0)
        
        # Generate class center
        center = torch.zeros(input_dim)
        center[0] = c * 2.0  # Separate classes along first dimension
        
        # Generate samples
        X_class = torch.randn(n_class, input_dim) * 0.5 + center
        y_class = torch.full((n_class,), c)
        
        X_list.append(X_class)
        y_list.append(y_class)
    
    # Combine
    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)
    
    # Shuffle
    perm = torch.randperm(n_samples)
    X = X[perm]
    y = y[perm]
    
    return X, y


def add_label_noise(labels: torch.Tensor, 
                   noise_prob: float = 0.1,
                   seed: Optional[int] = None) -> torch.Tensor:
    """
    Add random label noise to training data.
    
    Args:
        labels: Original labels
        noise_prob: Probability of flipping each label
        seed: Random seed for reproducibility
        
    Returns:
        Labels with added noise
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    noisy_labels = labels.clone()
    flip_mask = torch.rand(labels.shape[0]) < noise_prob
    noisy_labels[flip_mask] *= -1  # Flip binary labels
    
    return noisy_labels


class DataGenerator:
    """
    Flexible data generator for various experimental setups.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data generator with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
    def generate_dataset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate dataset according to configuration.
        
        Returns:
            Tuple of (features, labels)
        """
        data_type = self.config.get('type', 'gaussian')
        
        if data_type == 'gaussian':
            return generate_gaussian_data(
                n=self.config['n_samples'],
                d=self.config['input_dim'],
                separation=self.config.get('separation', 1.0),
                noise=self.config.get('noise', 0.1),
                seed=self.config.get('seed')
            )
        elif data_type == 'imbalanced':
            return generate_imbalanced_data(
                n_samples=self.config['n_samples'],
                input_dim=self.config['input_dim'],
                imbalance_ratio=self.config.get('imbalance_ratio', 0.1),
                seed=self.config.get('seed')
            )
        elif data_type == 'correlated':
            return generate_correlated_data(
                n_samples=self.config['n_samples'],
                input_dim=self.config['input_dim'],
                correlation_strength=self.config.get('correlation_strength', 0.5),
                seed=self.config.get('seed')
            )
        elif data_type == 'multiclass':
            return generate_multiclass_data(
                n_samples=self.config['n_samples'],
                input_dim=self.config['input_dim'],
                n_classes=self.config.get('n_classes', 3),
                seed=self.config.get('seed')
            )
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
    def get_data_statistics(self, features: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        Compute statistics about the generated data.
        
        Args:
            features: Input features
            labels: Target labels
            
        Returns:
            Dictionary of data statistics
        """
        stats = {
            'n_samples': features.shape[0],
            'n_features': features.shape[1],
            'feature_mean': torch.mean(features).item(),
            'feature_std': torch.std(features).item(),
            'label_balance': torch.mean((labels > 0).float()).item(),
            'feature_norm_mean': torch.mean(torch.norm(features, dim=1)).item(),
            'feature_norm_std': torch.std(torch.norm(features, dim=1)).item()
        }
        
        return stats
