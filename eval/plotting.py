"""
Plotting and Visualization Utilities for Mirror Descent Experiments.

This module provides functions for creating analysis plots and visualizations
from experiment results, including time series plots and aggregated comparisons.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import torch
import torch.nn as nn

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")


def plot_angles_over_time(results: Dict[str, List[Dict]], savepath: str, 
                         metric: str = 'angle_to_svm', title: str = None) -> None:
    """
    Plot angles over time with mean ± std across seeds.
    
    Args:
        results: Dictionary mapping experiment names to metrics history
        savepath: Path to save the plot
        metric: Metric to plot (e.g., 'angle_to_svm', 'angle_to_logreg')
        title: Plot title (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for exp_name, metrics_list in results.items():
        if not metrics_list:
            continue
            
        # Extract epochs and metric values
        epochs = [m.get('epoch', 0) for m in metrics_list]
        values = [m.get(metric, np.nan) for m in metrics_list]
        
        # Remove NaN values
        valid_indices = [i for i, v in enumerate(values) if not np.isnan(v)]
        epochs = [epochs[i] for i in valid_indices]
        values = [values[i] for i in valid_indices]
        
        if epochs and values:
            ax.plot(epochs, values, label=exp_name, linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel(f'Angle (degrees)')
    ax.set_title(title or f'{metric.replace("_", " ").title()} Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close()


def plot_margins_over_time(results: Dict[str, List[Dict]], savepath: str,
                          title: str = None) -> None:
    """
    Plot margins over time with mean ± std across seeds.
    
    Args:
        results: Dictionary mapping experiment names to metrics history
        savepath: Path to save the plot
        title: Plot title (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for exp_name, metrics_list in results.items():
        if not metrics_list:
            continue
            
        # Extract epochs and margin values
        epochs = [m.get('epoch', 0) for m in metrics_list]
        values = [m.get('margin', np.nan) for m in metrics_list]
        
        # Remove NaN values
        valid_indices = [i for i, v in enumerate(values) if not np.isnan(v)]
        epochs = [epochs[i] for i in valid_indices]
        values = [values[i] for i in valid_indices]
        
        if epochs and values:
            ax.plot(epochs, values, label=exp_name, linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Margin')
    ax.set_title(title or 'Margin Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close()


def plot_norm_balance(results: Dict[str, List[Dict]], savepath: str,
                     title: str = None) -> None:
    """
    Plot norm balance over time.
    
    Args:
        results: Dictionary mapping experiment names to metrics history
        savepath: Path to save the plot
        title: Plot title (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for exp_name, metrics_list in results.items():
        if not metrics_list:
            continue
            
        # Extract epochs and norm balance values
        epochs = [m.get('epoch', 0) for m in metrics_list]
        values = [m.get('norm_balance', np.nan) for m in metrics_list]
        
        # Remove NaN values
        valid_indices = [i for i, v in enumerate(values) if not np.isnan(v)]
        epochs = [epochs[i] for i in valid_indices]
        values = [values[i] for i in valid_indices]
        
        if epochs and values:
            ax.plot(epochs, values, label=exp_name, linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Norm Balance (max/min)')
    ax.set_title(title or 'Norm Balance Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale for norm balance
    
    plt.tight_layout()
    plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close()


def plot_decision_boundary(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
                          savepath: str, title: str = None) -> None:
    """
    Plot decision boundary for 2D synthetic data.
    
    Args:
        model: Trained model
        X: Input features (2D)
        y: Target labels
        savepath: Path to save the plot
        title: Plot title (optional)
    """
    model.eval()
    
    # Create grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    # Flatten grid
    grid_points = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()
    
    # Get model predictions
    with torch.no_grad():
        if hasattr(model, 'effective_weight'):
            u = model.effective_weight()
            model_pred = grid_points @ u
        else:
            model_pred = model(grid_points).squeeze()
        
        model_pred = torch.sign(model_pred).numpy()
    
    # Reshape for plotting
    model_pred = model_pred.reshape(xx.shape)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Decision boundary
    ax.contourf(xx, yy, model_pred, alpha=0.3, cmap='RdYlBu')
    
    # Data points
    pos_mask = y == 1
    neg_mask = y == -1
    
    ax.scatter(X[pos_mask, 0], X[pos_mask, 1], c='red', marker='o', 
               label='Class +1', s=50, alpha=0.8)
    ax.scatter(X[neg_mask, 0], X[neg_mask, 1], c='blue', marker='s', 
               label='Class -1', s=50, alpha=0.8)
    
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(title or 'Decision Boundary')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close()


def plot_aggregated_metrics(results_df: pd.DataFrame, savepath: str,
                           metric: str = 'angle_to_svm') -> None:
    """
    Plot aggregated metrics across different parameter combinations.
    
    Args:
        results_df: DataFrame with results from multiple experiments
        savepath: Path to save the plot
        metric: Metric to plot
    """
    if metric not in results_df.columns:
        print(f"Metric '{metric}' not found in results")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot by potential
    if 'potential' in results_df.columns:
        sns.boxplot(data=results_df, x='potential', y=metric, ax=axes[0])
        axes[0].set_title(f'{metric} by Potential')
        axes[0].tick_params(axis='x', rotation=45)
    
    # Plot by depth
    if 'depth' in results_df.columns:
        sns.boxplot(data=results_df, x='depth', y=metric, ax=axes[1])
        axes[1].set_title(f'{metric} by Depth')
    
    # Plot by p-value (if applicable)
    if 'p' in results_df.columns:
        sns.scatterplot(data=results_df, x='p', y=metric, hue='potential', ax=axes[2])
        axes[2].set_title(f'{metric} by P-value')
    
    # Plot margin vs angle relationship
    if 'final_margin' in results_df.columns:
        sns.scatterplot(data=results_df, x='final_margin', y=metric, 
                       hue='potential', ax=axes[3])
        axes[3].set_title(f'{metric} vs Final Margin')
    
    plt.tight_layout()
    plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close()


def plot_heatmap_metrics(results_df: pd.DataFrame, savepath: str,
                        x_col: str = 'potential', y_col: str = 'depth',
                        metric: str = 'angle_to_svm') -> None:
    """
    Create heatmap of metrics across parameter combinations.
    
    Args:
        results_df: DataFrame with results
        savepath: Path to save the plot
        x_col: Column for x-axis
        y_col: Column for y-axis
        metric: Metric to visualize
    """
    if any(col not in results_df.columns for col in [x_col, y_col, metric]):
        print(f"Required columns not found in results")
        return
    
    # Pivot data for heatmap
    pivot_data = results_df.pivot_table(
        values=metric, 
        index=y_col, 
        columns=x_col, 
        aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='viridis', ax=ax)
    ax.set_title(f'{metric.replace("_", " ").title()} Heatmap')
    ax.set_xlabel(x_col.replace('_', ' ').title())
    ax.set_ylabel(y_col.replace('_', ' ').title())
    
    plt.tight_layout()
    plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_curves(metrics_history: List[Dict], savepath: str) -> None:
    """
    Plot training curves from metrics history.
    
    Args:
        metrics_history: List of metric dictionaries
        savepath: Path to save the plot
    """
    if not metrics_history:
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(metrics_history)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    if 'train_loss' in df.columns:
        axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train Loss')
        if 'val_loss' in df.columns:
            axes[0, 0].plot(df['epoch'], df['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy curves
    if 'train_accuracy' in df.columns:
        axes[0, 1].plot(df['epoch'], df['train_accuracy'], label='Train Acc')
        if 'val_accuracy' in df.columns:
            axes[0, 1].plot(df['epoch'], df['val_accuracy'], label='Val Acc')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Margin curve
    if 'margin' in df.columns:
        axes[1, 0].plot(df['epoch'], df['margin'])
        axes[1, 0].set_title('Margin Over Time')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Margin')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Weight norm curve
    if 'effective_weight_norm' in df.columns:
        axes[1, 1].plot(df['epoch'], df['effective_weight_norm'])
        axes[1, 1].set_title('Weight Norm Over Time')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Weight Norm')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close()


def create_publication_plots(results_df: pd.DataFrame, output_dir: str) -> None:
    """
    Create publication-ready plots from aggregated results.
    
    Args:
        results_df: DataFrame with aggregated results
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set publication style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    })
    
    # Plot 1: Angles by potential and depth
    if 'angle_to_svm' in results_df.columns:
        plot_heatmap_metrics(
            results_df, 
            output_dir / "angles_heatmap.png",
            x_col='potential', 
            y_col='depth',
            metric='angle_to_svm'
        )
    
    # Plot 2: Margin vs angle relationship
    if 'final_margin' in results_df.columns and 'angle_to_svm' in results_df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            data=results_df, 
            x='final_margin', 
            y='angle_to_svm', 
            hue='potential',
            size='depth',
            sizes=(50, 200),
            ax=ax
        )
        ax.set_xlabel('Final Margin')
        ax.set_ylabel('Angle to SVM (degrees)')
        ax.set_title('Margin vs Angle Relationship')
        plt.tight_layout()
        plt.savefig(output_dir / "margin_vs_angle.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 3: Norm balance by potential
    if 'norm_balance' in results_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=results_df, x='potential', y='norm_balance', ax=ax)
        ax.set_yscale('log')
        ax.set_ylabel('Norm Balance (log scale)')
        ax.set_title('Norm Balance by Potential Function')
        plt.tight_layout()
        plt.savefig(output_dir / "norm_balance_by_potential.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Publication plots saved to {output_dir}")
