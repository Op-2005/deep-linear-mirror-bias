"""
Core module for Mirror Descent implicit bias experiments.

This module contains the fundamental components for studying implicit bias
in deep linear networks trained with Mirror Descent optimization.
"""

from .models import DeepLinear, Linear
from .md_optimizer import MirrorDescentOptimizer, AdaptiveMirrorDescent
from .potentials import (
    QuadraticPotential, 
    LpPotential, 
    ScaledPotential, 
    EntropyPotential,
    LayerScaledQuadratic,
    test_potential_consistency,
    run_potential_tests
)

__all__ = [
    'DeepLinear',
    'Linear', 
    'MirrorDescentOptimizer',
    'AdaptiveMirrorDescent',
    'QuadraticPotential',
    'LpPotential', 
    'ScaledPotential',
    'EntropyPotential',
    'LayerScaledQuadratic',
    'test_potential_consistency',
    'run_potential_tests'
]
