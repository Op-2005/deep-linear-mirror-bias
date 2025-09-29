"""
Training Script for Mirror Descent Experiments.

This script provides the main entry point for training deep linear networks
with Mirror Descent optimization on synthetic data or MNIST and analyzing their implicit bias properties.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List

from core.models import Linear, DeepLinear
from core.md_optimizer import MirrorDescentOptimizer
from core.potentials import (
    QuadraticPotential, 
    LpPotential, 
    LayerScaledQuadratic, 
    ScaledPotential
)
from data.synthetic import generate_gaussian_data, standardize, train_val_split
from data.mnist import load_mnist_binary
from eval.baselines import svm_l2, logreg_gd
from eval.metrics import (
    margin, 
    angles_to_baselines, 
    layer_alignment, 
    norm_balance,
    analyze_implicit_bias
)
from utils.experiment import (
    make_experiment_dir,
    save_json,
    save_yaml,
    set_seeds,
    save_experiment_config,
    save_metrics_history,
    save_metrics_csv,
    create_experiment_summary,
    get_version_stamp
)


def create_potential(potential_name: str, **kwargs) -> Any:
    """
    Create potential function based on name and parameters.
    
    Args:
        potential_name: Name of the potential function
        **kwargs: Additional parameters for the potential
        
    Returns:
        Potential function instance
    """
    if potential_name == 'quadratic':
        return QuadraticPotential()
    elif potential_name == 'lp':
        p = kwargs.get('p', 2.0)
        return LpPotential(p=p)
    elif potential_name == 'layer_scaled':
        alpha = kwargs.get('alpha', 1.0)
        return LayerScaledQuadratic(alpha=alpha)
    elif potential_name == 'scaled':
        return ScaledPotential()
    else:
        raise ValueError(f"Unknown potential: {potential_name}")


def train_model(model: nn.Module, 
                optimizer: MirrorDescentOptimizer,
                X_train: torch.Tensor,
                y_train: torch.Tensor,
                X_val: torch.Tensor,
                y_val: torch.Tensor,
                epochs: int,
                criterion: nn.Module,
                baselines: Dict[str, torch.Tensor],
                log_frequency: int = 10) -> List[Dict[str, Any]]:
    """
    Train model with Mirror Descent optimization and log metrics over epochs.
    
    Args:
        model: Model to train
        optimizer: Mirror Descent optimizer
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        epochs: Number of training epochs
        criterion: Loss function
        baselines: Dictionary of baseline weight vectors
        log_frequency: Frequency of logging metrics
        
    Returns:
        List of metric dictionaries for each logged epoch
    """
    metrics_history = []
    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_train)
        # Convert targets from [-1, 1] to [0, 1] for BCE loss
        y_train_bce = (y_train + 1) / 2.0
        loss = criterion(outputs.squeeze(), y_train_bce)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Log metrics at specified frequency
        if epoch % log_frequency == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                # Basic metrics
                epoch_metrics = {
                    'epoch': epoch,
                    'train_loss': loss.item(),
                    'train_accuracy': torch.mean((torch.sign(outputs.squeeze()) == y_train).float()).item()
                }
                
                # Validation metrics
                val_outputs = model(X_val)
                # Convert validation targets from [-1, 1] to [0, 1] for BCE loss
                y_val_bce = (y_val + 1) / 2.0
                val_loss = criterion(val_outputs.squeeze(), y_val_bce)
                epoch_metrics['val_loss'] = val_loss.item()
                epoch_metrics['val_accuracy'] = torch.mean((torch.sign(val_outputs.squeeze()) == y_val).float()).item()
                
                # Model-specific metrics
                if hasattr(model, 'effective_weight'):
                    u = model.effective_weight()
                    epoch_metrics['effective_weight_norm'] = torch.norm(u).item()
                    epoch_metrics['margin'] = margin(u, X_train, y_train).item()
                    
                    # Angles to baselines
                    angles = angles_to_baselines(u, X_train, y_train, baselines)
                    epoch_metrics.update(angles)
                    
                    # Layer-specific metrics for deep networks
                    if hasattr(model, 'get_layer_weights'):
                        W_list = model.get_layer_weights()
                        epoch_metrics['layer_alignment'] = layer_alignment(W_list)
                        epoch_metrics['norm_balance'] = norm_balance(W_list)
                
                metrics_history.append(epoch_metrics)
            
            model.train()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    
    return metrics_history


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Mirror Descent models on synthetic or MNIST data')
    
    # Model arguments
    parser.add_argument('--potential', type=str, default='quadratic',
                       choices=['quadratic', 'lp', 'layer_scaled', 'scaled'],
                       help='Potential function to use')
    parser.add_argument('--p', type=float, default=2.0,
                       help='Order of Lp norm (for lp potential)')
    parser.add_argument('--L', type=int, default=2,
                       help='Number of hidden layers (depth)')
    
    # Data arguments
    parser.add_argument('--dataset', type=str, default='synthetic',
                       choices=['synthetic', 'mnist'],
                       help='Dataset to use')
    parser.add_argument('--input_dim', type=int, default=2,
                       help='Input dimension (for synthetic data)')
    parser.add_argument('--n_samples', type=int, default=512,
                       help='Number of training samples (for synthetic data)')
    parser.add_argument('--separation', type=float, default=3.0,
                       help='Separation between class centers (for synthetic data)')
    parser.add_argument('--noise', type=float, default=0.2,
                       help='Noise level (for synthetic data)')
    parser.add_argument('--digits', type=int, nargs=2, default=[0, 1],
                       help='Two digits for MNIST binary classification')
    parser.add_argument('--pca', type=int, default=None,
                       help='PCA dimensions for MNIST (optional)')
    
    # Training arguments
    parser.add_argument('--lr', type=float, default=0.1,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=1000,
                       help='Number of training epochs')
    parser.add_argument('--normalize_md', action='store_true',
                       help='Normalize Mirror Descent updates')
    parser.add_argument('--log_frequency', type=int, default=10,
                       help='Frequency of logging metrics')
    
    # Experiment management
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed')
    parser.add_argument('--output_base', type=str, default='runs_out',
                       help='Base directory for experiment outputs')
    parser.add_argument('--tag', type=str, default=None,
                       help='Tag for experiment directory')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    set_seeds(args.seed)
    
    # Create experiment directory
    exp_dir = make_experiment_dir(base=args.output_base, tag=args.tag)
    
    # Save configuration with version stamp
    config = vars(args)
    config['version_stamp'] = get_version_stamp()
    save_experiment_config(config, exp_dir)
    
    print(f"Starting experiment with potential={args.potential}, L={args.L}")
    print(f"Experiment directory: {exp_dir}")
    
    # Load data based on dataset type
    if args.dataset == 'synthetic':
        print(f"Generating synthetic data: input_dim={args.input_dim}, n_samples={args.n_samples}")
        
        # Generate synthetic data
        X, y = generate_gaussian_data(
            n=args.n_samples,
            d=args.input_dim,
            separation=args.separation,
            noise=args.noise,
            seed=args.seed
        )
        
        # Standardize features
        X, mean, std = standardize(X)
        
        # Split into train/val
        X_train, y_train, X_val, y_val = train_val_split(X, y, val_ratio=0.2, seed=args.seed)
        
    elif args.dataset == 'mnist':
        print(f"Loading MNIST data: digits {args.digits[0]} vs {args.digits[1]}")
        
        # Load MNIST data
        X_train, y_train, X_val, y_val = load_mnist_binary(
            d1=args.digits[0],
            d2=args.digits[1],
            flatten=True,
            pca_components=args.pca,
            standardize=True,
            seed=args.seed
        )
        
        # Use validation set as test set for MNIST
        X_test, y_test = X_val, y_val
        # Create a smaller validation set from training data
        X_train, y_train, X_val, y_val = train_val_split(X_train, y_train, val_ratio=0.2, seed=args.seed)
    
    print(f"Data: {X_train.shape[0]} train samples, {X_val.shape[0]} val samples")
    
    # Get input dimension for model creation
    input_dim = X_train.shape[1]
    
    # Create potential function
    potential_kwargs = {}
    if args.potential == 'lp':
        potential_kwargs['p'] = args.p
    elif args.potential == 'layer_scaled':
        potential_kwargs['alpha'] = 1.0
    
    potential = create_potential(args.potential, **potential_kwargs)
    
    # Create models
    if args.L == 1:
        # Linear model
        model = Linear(input_dim, output_dim=1, bias=False)
        hidden_dims = []
    else:
        # Deep linear model
        hidden_dims = [50] * (args.L - 1)  # All hidden layers have same width
        model = DeepLinear(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            bias=False
        )
    
    # Create optimizer
    optimizer = MirrorDescentOptimizer(
        model.parameters(),
        potentials=potential,
        learning_rate=args.lr,
        normalize_md=args.normalize_md
    )
    
    # Learning rate sanity checks
    if args.potential == 'lp' and args.p != 2.0:
        # Lp potentials (p≠2) are more sensitive around 0
        if args.lr > 0.1 and not args.normalize_md:
            print(f"WARNING: Lp potential (p={args.p}) with high LR ({args.lr}) without normalization.")
            print("Consider using --normalize_md flag or reducing learning rate.")
        elif args.lr > 0.05:
            print(f"INFO: Using LR {args.lr} for Lp potential (p={args.p}). Consider --normalize_md for stability.")
    
    if args.potential == 'scaled':
        print("INFO: Using 'scaled' potential with dual clipping enabled for stability.")
    
    # Loss function - use BCE with logits for binary classification
    criterion = nn.BCEWithLogitsLoss()
    
    print(f"Training {type(model).__name__} with {args.potential} potential...")
    
    # Fit baselines first
    print("Fitting baseline methods...")
    svm_weight = svm_l2(X_train, y_train)
    logreg_weight = logreg_gd(X_train, y_train)
    baselines = {'svm': svm_weight, 'logreg': logreg_weight}
    
    # Train model with logging
    metrics_history = train_model(
        model, optimizer, X_train, y_train, X_val, y_val, 
        args.epochs, criterion, baselines, args.log_frequency
    )
    
    # Compute final metrics
    model.eval()
    with torch.no_grad():
        train_pred = model(X_train)
        train_acc = torch.mean((torch.sign(train_pred.squeeze()) == y_train).float()).item()
        
        val_pred = model(X_val)
        val_acc = torch.mean((torch.sign(val_pred.squeeze()) == y_val).float()).item()
    
    print(f"Final accuracy - Train: {train_acc:.4f}, Val: {val_acc:.4f}")
    
    # Compute final metrics
    final_metrics = {
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'potential': args.potential,
        'depth': args.L,
        'input_dim': input_dim,
        'lr': args.lr,
        'epochs': args.epochs,
        'seed': args.seed,
        'dataset': args.dataset
    }
    
    # Add dataset-specific parameters
    if args.dataset == 'synthetic':
        final_metrics.update({
            'n_samples': args.n_samples,
            'separation': args.separation,
            'noise': args.noise
        })
    elif args.dataset == 'mnist':
        final_metrics.update({
            'digits': args.digits,
            'pca_components': args.pca
        })
    
    # Add model-specific metrics
    if hasattr(model, 'effective_weight'):
        u = model.effective_weight()
        final_metrics['final_margin'] = margin(u, X_train, y_train).item()
        final_metrics['final_weight_norm'] = torch.norm(u).item()
        
        # Angles to baselines
        angles = angles_to_baselines(u, X_train, y_train, baselines)
        final_metrics.update(angles)
        
        # Layer-specific metrics for deep networks
        if hasattr(model, 'get_layer_weights'):
            W_list = model.get_layer_weights()
            final_metrics['layer_alignment'] = layer_alignment(W_list)
            final_metrics['norm_balance'] = norm_balance(W_list)
    
    # Save results
    save_json(final_metrics, Path(exp_dir) / "results.json")
    
    # Save metrics history
    save_metrics_history(metrics_history, exp_dir)
    save_metrics_csv(metrics_history, exp_dir)
    
    # Create decision boundary plot for 2D synthetic data
    if args.dataset == 'synthetic' and input_dim == 2:
        plot_decision_boundary(model, X_train, y_train, X_val, y_val, 
                              svm_weight, logreg_weight, exp_dir)
    
    # Create experiment summary
    create_experiment_summary(exp_dir, final_metrics)
    
    print(f"Experiment completed successfully! Results saved to {exp_dir}")


def plot_decision_boundary(model: nn.Module, 
                          X_train: torch.Tensor, 
                          y_train: torch.Tensor,
                          X_val: torch.Tensor,
                          y_val: torch.Tensor,
                          svm_weight: torch.Tensor,
                          logreg_weight: torch.Tensor,
                          output_dir: str):
    """
    Plot decision boundary for 2D data.
    
    Args:
        model: Trained model
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        svm_weight: SVM weight vector
        logreg_weight: Logistic regression weight vector
        output_dir: Output directory
    """
    model.eval()
    
    # Create grid
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    # Flatten grid
    grid_points = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()
    
    # Get model predictions
    with torch.no_grad():
        if hasattr(model, 'effective_weight'):
            u = model.effective_weight()
            u = u.flatten()  # Ensure u is the right shape
            model_pred = grid_points @ u
        else:
            model_pred = model(grid_points).squeeze()
        
        model_pred = torch.sign(model_pred).numpy()
    
    # Reshape for plotting
    model_pred = model_pred.reshape(xx.shape)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Model decision boundary
    axes[0].contourf(xx, yy, model_pred, alpha=0.3, cmap='RdYlBu')
    axes[0].scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], 
                   c='red', marker='o', label='Train +1')
    axes[0].scatter(X_train[y_train == -1, 0], X_train[y_train == -1, 1], 
                   c='blue', marker='s', label='Train -1')
    axes[0].scatter(X_val[y_val == 1, 0], X_val[y_val == 1, 1], 
                   c='darkred', marker='^', label='Val +1')
    axes[0].scatter(X_val[y_val == -1, 0], X_val[y_val == -1, 1], 
                   c='darkblue', marker='v', label='Val -1')
    axes[0].set_title('Deep Linear Network')
    axes[0].legend()
    
    # SVM decision boundary
    svm_pred = grid_points @ svm_weight
    svm_pred = torch.sign(svm_pred).numpy().reshape(xx.shape)
    axes[1].contourf(xx, yy, svm_pred, alpha=0.3, cmap='RdYlBu')
    axes[1].scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], 
                   c='red', marker='o', label='Train +1')
    axes[1].scatter(X_train[y_train == -1, 0], X_train[y_train == -1, 1], 
                   c='blue', marker='s', label='Train -1')
    axes[1].set_title('SVM')
    axes[1].legend()
    
    # Logistic regression decision boundary
    logreg_pred = grid_points @ logreg_weight
    logreg_pred = torch.sign(logreg_pred).numpy().reshape(xx.shape)
    axes[2].contourf(xx, yy, logreg_pred, alpha=0.3, cmap='RdYlBu')
    axes[2].scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], 
                   c='red', marker='o', label='Train +1')
    axes[2].scatter(X_train[y_train == -1, 0], X_train[y_train == -1, 1], 
                   c='blue', marker='s', label='Train -1')
    axes[2].set_title('Logistic Regression')
    axes[2].legend()
    
    plt.tight_layout()
    plot_path = Path(output_dir) / "figs" / "decision_boundary.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
