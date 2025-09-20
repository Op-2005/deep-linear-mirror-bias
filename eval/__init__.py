"""
Evaluation module for Mirror Descent experiments.

This module provides utilities for evaluating models and comparing
their implicit bias properties across different optimization methods.
"""

from .metrics import (
    margin,
    angle,
    angles_to_baselines,
    layer_alignment,
    norm_balance,
    ntk_drift,
    compute_spectral_properties,
    analyze_implicit_bias,
    MetricsCollector
)
from .baselines import (
    solve_svm,
    solve_p_margin,
    solve_logistic_regression,
    solve_ridge_regression,
    solve_lasso_regression,
    solve_elastic_net,
    compare_baselines,
    BaselineRunner
)

__all__ = [
    'compute_margin',
    'compute_angles',
    'compute_alignment', 
    'compute_spectral_properties',
    'analyze_implicit_bias',
    'MetricsCollector',
    'solve_svm',
    'solve_p_margin',
    'solve_logistic_regression',
    'solve_ridge_regression',
    'solve_lasso_regression',
    'solve_elastic_net',
    'compare_baselines',
    'BaselineRunner'
]
