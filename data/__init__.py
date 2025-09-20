"""
Data module for Mirror Descent experiments.

This module provides utilities for loading and generating datasets
for studying implicit bias in deep linear networks.
"""

from .synthetic import (
    generate_gaussian_data,
    generate_imbalanced_data,
    generate_correlated_data,
    generate_multiclass_data,
    DataGenerator
)
try:
    from .mnist import (
        load_mnist_binary,
        preprocess_mnist,
        create_mnist_subset,
        MNISTDataManager
    )
except ImportError:
    # MNIST functionality not available
    pass

__all__ = [
    'generate_gaussian_data',
    'generate_imbalanced_data', 
    'generate_correlated_data',
    'generate_multiclass_data',
    'DataGenerator',
    'load_mnist_binary',
    'preprocess_mnist',
    'create_mnist_subset',
    'MNISTDataManager'
]
