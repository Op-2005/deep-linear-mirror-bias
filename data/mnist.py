"""
MNIST Data Loading for Mirror Descent Experiments.

This module provides utilities for loading and preprocessing MNIST data
for binary classification tasks in Mirror Descent experiments.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional, Dict, Any
import numpy as np


def load_mnist_binary(d1: int = 0, 
                     d2: int = 1,
                     flatten: bool = True, 
                     pca_components: Optional[int] = None, 
                     standardize: bool = True, 
                     seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load MNIST data for binary classification.
    
    Args:
        d1: First digit class (0-9)
        d2: Second digit class (0-9)
        flatten: Whether to flatten images to vectors
        pca_components: Optional PCA dimensionality reduction
        standardize: Whether to standardize features
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, y_train, X_test, y_test) with labels ±1
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Download MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Filter for binary classification
    train_mask = (train_dataset.targets == d1) | (train_dataset.targets == d2)
    test_mask = (test_dataset.targets == d1) | (test_dataset.targets == d2)
    
    train_data = train_dataset.data[train_mask].float()
    train_labels = train_dataset.targets[train_mask]
    test_data = test_dataset.data[test_mask].float()
    test_labels = test_dataset.targets[test_mask]
    
    # Convert to binary labels: d1 -> +1, d2 -> -1
    train_labels = torch.where(train_labels == d1, 1, -1)
    test_labels = torch.where(test_labels == d1, 1, -1)
    
    # Flatten if requested
    if flatten:
        train_data = train_data.view(train_data.size(0), -1)
        test_data = test_data.view(test_data.size(0), -1)
    
    # Apply PCA if requested
    if pca_components is not None:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=pca_components)
        
        # Fit PCA on training data
        train_data_np = train_data.numpy()
        pca.fit(train_data_np)
        
        # Transform both train and test
        train_data = torch.from_numpy(pca.transform(train_data_np)).float()
        test_data_np = test_data.numpy()
        test_data = torch.from_numpy(pca.transform(test_data_np)).float()
    
    # Standardize if requested
    if standardize:
        # Compute mean and std on training data
        train_mean = torch.mean(train_data, dim=0)
        train_std = torch.std(train_data, dim=0)
        train_std = torch.clamp(train_std, min=1e-8)
        
        # Apply standardization
        train_data = (train_data - train_mean) / train_std
        test_data = (test_data - train_mean) / train_std
    
    return train_data, train_labels, test_data, test_labels


def preprocess_mnist(images: torch.Tensor,
                    labels: torch.Tensor,
                    normalize: bool = True,
                    flatten: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocess MNIST images and labels.
    
    Args:
        images: Raw MNIST images
        labels: Raw MNIST labels
        normalize: Whether to normalize pixel values
        flatten: Whether to flatten images to vectors
        
    Returns:
        Tuple of (processed_features, processed_labels)
    """
    processed_images = images.float()
    
    if flatten:
        processed_images = processed_images.view(processed_images.size(0), -1)
    
    if normalize:
        # Normalize to [0, 1] range
        processed_images = processed_images / 255.0
        
        # Standardize to zero mean and unit variance
        mean = torch.mean(processed_images, dim=0)
        std = torch.std(processed_images, dim=0)
        std = torch.clamp(std, min=1e-8)
        processed_images = (processed_images - mean) / std
    
    return processed_images, labels


def create_mnist_subset(features: torch.Tensor,
                       labels: torch.Tensor,
                       n_samples_per_class: int = 1000,
                       seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a subset of MNIST data for faster experiments.
    
    Args:
        features: MNIST features
        labels: MNIST labels
        n_samples_per_class: Number of samples per class to keep
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (subset_features, subset_labels)
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Find indices for each class
    pos_indices = torch.where(labels == 1)[0]
    neg_indices = torch.where(labels == -1)[0]
    
    # Sample subset from each class
    n_pos = min(n_samples_per_class, len(pos_indices))
    n_neg = min(n_samples_per_class, len(neg_indices))
    
    pos_subset = torch.randperm(len(pos_indices))[:n_pos]
    neg_subset = torch.randperm(len(neg_indices))[:n_neg]
    
    # Combine subsets
    subset_indices = torch.cat([
        pos_indices[pos_subset],
        neg_indices[neg_subset]
    ])
    
    # Shuffle
    perm = torch.randperm(len(subset_indices))
    subset_indices = subset_indices[perm]
    
    return features[subset_indices], labels[subset_indices]


def get_mnist_data_loader(features: torch.Tensor,
                         labels: torch.Tensor,
                         batch_size: int = 32,
                         shuffle: bool = True) -> DataLoader:
    """
    Create a PyTorch DataLoader for MNIST data.
    
    Args:
        features: MNIST features
        labels: MNIST labels
        batch_size: Batch size for the data loader
        shuffle: Whether to shuffle the data
        
    Returns:
        PyTorch DataLoader
    """
    dataset = TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class MNISTDataManager:
    """
    Manager class for MNIST data operations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MNIST data manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
    def load_dataset(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load MNIST dataset according to configuration.
        
        Returns:
            Tuple of (train_features, train_labels, test_features, test_labels)
        """
        return load_mnist_binary(
            d1=self.config.get('digit1', 0),
            d2=self.config.get('digit2', 1),
            flatten=self.config.get('flatten', True),
            pca_components=self.config.get('pca_components'),
            standardize=self.config.get('standardize', True),
            seed=self.config.get('seed', 0)
        )
        
    def get_data_loaders(self, batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
        """
        Get training and testing data loaders.
        
        Args:
            batch_size: Batch size for data loaders
            
        Returns:
            Tuple of (train_loader, test_loader)
        """
        X_train, y_train, X_test, y_test = self.load_dataset()
        
        train_loader = get_mnist_data_loader(X_train, y_train, batch_size, shuffle=True)
        test_loader = get_mnist_data_loader(X_test, y_test, batch_size, shuffle=False)
        
        return train_loader, test_loader
        
    def augment_data(self, features: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply data augmentation to MNIST images.
        
        Args:
            features: Input features
            labels: Target labels
            
        Returns:
            Tuple of (augmented_features, augmented_labels)
        """
        # Simple augmentation: add small amount of noise
        noise_std = self.config.get('augment_noise', 0.01)
        noise = torch.randn_like(features) * noise_std
        augmented_features = features + noise
        
        return augmented_features, labels
