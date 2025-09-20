"""
Deep Linear Network Models for Mirror Descent Implicit Bias Analysis.

This module implements the core model architectures used in the study of implicit bias
in deep linear networks trained with Mirror Descent optimization.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class Linear(nn.Module):
    """
    Single linear layer for baseline comparisons.
    
    This serves as the shallow baseline to compare against deep linear networks
    in terms of implicit bias properties.
    """
    
    def __init__(self, input_dim: int, output_dim: int = 1, bias: bool = True):
        """
        Initialize linear layer.
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension (default: 1 for binary classification)
            bias: Whether to include bias term
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through linear layer.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        return self.linear(x)
    
    def get_layer_weights(self) -> List[torch.Tensor]:
        """
        Return only weight tensors (ignoring biases).
        
        Returns:
            List containing the weight tensor
        """
        return [self.linear.weight.data]


class DeepLinear(nn.Module):
    """
    Deep linear network without nonlinearities.
    
    A composition of linear layers that forms the main architecture
    for studying implicit bias in deep networks. The key insight is that
    even without nonlinearities, depth can induce different implicit bias
    compared to shallow networks.
    """
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dims: List[int],
                 output_dim: int = 1,
                 bias: bool = True):
        """
        Initialize deep linear network.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output feature dimension (default: 1 for binary classification)
            bias: Whether to include bias terms
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.bias = bias
        
        # Build layers
        dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i+1], bias=bias) 
            for i in range(len(dims)-1)
        ])
        
        # Cache for effective weight
        self._effective_weight_cache = None
        self._cache_valid = False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through deep linear network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        for layer in self.layers:
            x = layer(x)
        return x
        
    def get_feature_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract feature maps at each layer for analysis.
        
        Args:
            x: Input tensor
            
        Returns:
            List of feature maps at each layer
        """
        feature_maps = [x]
        for layer in self.layers:
            x = layer(x)
            feature_maps.append(x)
        return feature_maps
    
    def effective_weight(self, cache: bool = True) -> torch.Tensor:
        """
        Multiply weights right-to-left into a single vector/matrix u.
        Cache the result and invalidate on .train() or weight changes.
        
        Args:
            cache: Whether to use cached result if available
            
        Returns:
            Effective weight matrix/vector
        """
        if cache and self._cache_valid and self._effective_weight_cache is not None:
            return self._effective_weight_cache
            
        # Compute effective weight by multiplying from right to left
        effective_w = self.layers[-1].weight.data
        for layer in reversed(self.layers[:-1]):
            effective_w = effective_w @ layer.weight.data
            
        if cache:
            self._effective_weight_cache = effective_w
            self._cache_valid = True
            
        return effective_w
    
    def get_layer_weights(self) -> List[torch.Tensor]:
        """
        Return only weight tensors (ignoring biases).
        
        Returns:
            List of weight tensors for each layer
        """
        return [layer.weight.data for layer in self.layers]
    
    def train(self, mode: bool = True):
        """
        Override train to invalidate effective weight cache.
        """
        super().train(mode)
        self._cache_valid = False
        self._effective_weight_cache = None
