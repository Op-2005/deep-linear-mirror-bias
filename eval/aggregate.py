"""
Result Aggregation Utilities for Mirror Descent Experiments.

This module provides functions for aggregating results from multiple experiments,
computing summary statistics, and generating analysis tables.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import json
import glob

from utils.experiment import load_json


def load_experiment_results(experiment_dirs: List[str]) -> pd.DataFrame:
    """
    Load results from multiple experiment directories.
    
    Args:
        experiment_dirs: List of experiment directory paths
        
    Returns:
        DataFrame with all results
    """
    results = []
    
    for exp_dir in experiment_dirs:
        results_file = Path(exp_dir) / "results.json"
        if results_file.exists():
            try:
                result_data = load_json(results_file)
                result_data['experiment_dir'] = exp_dir
                results.append(result_data)
            except Exception as e:
                print(f"Failed to load results from {results_file}: {e}")
    
    return pd.DataFrame(results)


def aggregate_by_parameters(results_df: pd.DataFrame, 
                          groupby_cols: List[str] = None) -> pd.DataFrame:
    """
    Aggregate results by parameter groups.
    
    Args:
        results_df: DataFrame with experiment results
        groupby_cols: Columns to group by (default: potential, depth, p)
        
    Returns:
        Aggregated DataFrame with summary statistics
    """
    if groupby_cols is None:
        groupby_cols = ['potential', 'depth']
        # Add 'p' if it exists in the data
        if 'p' in results_df.columns:
            groupby_cols.append('p')
    
    # Only use columns that exist
    groupby_cols = [col for col in groupby_cols if col in results_df.columns]
    
    if not groupby_cols:
        print("No valid grouping columns found")
        return results_df
    
    # Define metrics to aggregate
    metrics_to_aggregate = [
        'train_accuracy', 'val_accuracy', 'final_margin', 'final_weight_norm',
        'angle_to_svm', 'angle_to_logreg', 'layer_alignment', 'norm_balance'
    ]
    
    # Only aggregate metrics that exist
    metrics_to_aggregate = [col for col in metrics_to_aggregate if col in results_df.columns]
    
    if not metrics_to_aggregate:
        print("No metrics to aggregate found")
        return results_df
    
    # Aggregate with multiple statistics
    agg_dict = {}
    for metric in metrics_to_aggregate:
        agg_dict[metric] = ['mean', 'std', 'min', 'max', 'count']
    
    # Add seed count
    if 'seed' in results_df.columns:
        agg_dict['seed'] = 'nunique'
    
    aggregated = results_df.groupby(groupby_cols).agg(agg_dict)
    
    # Flatten column names
    aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns.values]
    
    return aggregated


def compute_angle_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics for angles to baselines.
    
    Args:
        results_df: DataFrame with experiment results
        
    Returns:
        DataFrame with angle summaries
    """
    angle_cols = [col for col in results_df.columns if col.startswith('angle_to_')]
    
    if not angle_cols:
        print("No angle metrics found")
        return pd.DataFrame()
    
    summary = results_df[angle_cols].describe()
    
    # Add additional statistics
    for col in angle_cols:
        summary.loc['median', col] = results_df[col].median()
        summary.loc['q25', col] = results_df[col].quantile(0.25)
        summary.loc['q75', col] = results_df[col].quantile(0.75)
    
    return summary


def compute_margin_analysis(results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute margin analysis statistics.
    
    Args:
        results_df: DataFrame with experiment results
        
    Returns:
        Dictionary with margin analysis
    """
    if 'final_margin' not in results_df.columns:
        return {}
    
    analysis = {
        'margin_stats': results_df['final_margin'].describe().to_dict(),
        'margin_by_potential': results_df.groupby('potential')['final_margin'].describe() if 'potential' in results_df.columns else None,
        'margin_by_depth': results_df.groupby('depth')['final_margin'].describe() if 'depth' in results_df.columns else None,
    }
    
    return analysis


def compute_alignment_metrics(results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute layer alignment and norm balance metrics.
    
    Args:
        results_df: DataFrame with experiment results
        
    Returns:
        Dictionary with alignment metrics
    """
    metrics = {}
    
    if 'layer_alignment' in results_df.columns:
        metrics['layer_alignment'] = {
            'mean': results_df['layer_alignment'].mean(),
            'std': results_df['layer_alignment'].std(),
            'by_potential': results_df.groupby('potential')['layer_alignment'].mean() if 'potential' in results_df.columns else None,
            'by_depth': results_df.groupby('depth')['layer_alignment'].mean() if 'depth' in results_df.columns else None
        }
    
    if 'norm_balance' in results_df.columns:
        metrics['norm_balance'] = {
            'mean': results_df['norm_balance'].mean(),
            'std': results_df['norm_balance'].std(),
            'median': results_df['norm_balance'].median(),
            'by_potential': results_df.groupby('potential')['norm_balance'].mean() if 'potential' in results_df.columns else None,
            'by_depth': results_df.groupby('depth')['norm_balance'].mean() if 'depth' in results_df.columns else None
        }
    
    return metrics


def generate_summary_table(results_df: pd.DataFrame, 
                          output_path: str,
                          format: str = 'csv') -> None:
    """
    Generate summary table from aggregated results.
    
    Args:
        results_df: DataFrame with results
        output_path: Path to save summary table
        format: Output format ('csv' or 'latex')
    """
    # Aggregate by parameters
    aggregated = aggregate_by_parameters(results_df)
    
    if format == 'csv':
        aggregated.to_csv(output_path)
    elif format == 'latex':
        latex_table = aggregated.to_latex(
            float_format='%.3f',
            caption='Summary of Mirror Descent Experiments',
            label='tab:md_summary'
        )
        with open(output_path, 'w') as f:
            f.write(latex_table)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Summary table saved to {output_path}")


def compare_potentials(results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compare different potential functions.
    
    Args:
        results_df: DataFrame with experiment results
        
    Returns:
        Dictionary with potential comparison
    """
    if 'potential' not in results_df.columns:
        return {}
    
    comparison = {}
    
    # Key metrics to compare
    metrics = ['final_margin', 'angle_to_svm', 'angle_to_logreg', 'norm_balance']
    metrics = [m for m in metrics if m in results_df.columns]
    
    for metric in metrics:
        comparison[metric] = results_df.groupby('potential')[metric].agg(['mean', 'std', 'count'])
    
    return comparison


def compare_depths(results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compare different network depths.
    
    Args:
        results_df: DataFrame with experiment results
        
    Returns:
        Dictionary with depth comparison
    """
    if 'depth' not in results_df.columns:
        return {}
    
    comparison = {}
    
    # Key metrics to compare
    metrics = ['final_margin', 'angle_to_svm', 'angle_to_logreg', 'layer_alignment', 'norm_balance']
    metrics = [m for m in metrics if m in results_df.columns]
    
    for metric in metrics:
        comparison[metric] = results_df.groupby('depth')[metric].agg(['mean', 'std', 'count'])
    
    return comparison


def generate_comprehensive_report(results_df: pd.DataFrame, output_dir: str) -> None:
    """
    Generate comprehensive analysis report.
    
    Args:
        results_df: DataFrame with experiment results
        output_dir: Directory to save report files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Summary statistics
    summary_stats = results_df.describe()
    summary_stats.to_csv(output_dir / "summary_statistics.csv")
    
    # 2. Aggregated results
    aggregated = aggregate_by_parameters(results_df)
    aggregated.to_csv(output_dir / "aggregated_results.csv")
    
    # 3. Angle analysis
    angle_summary = compute_angle_summary(results_df)
    if not angle_summary.empty:
        angle_summary.to_csv(output_dir / "angle_analysis.csv")
    
    # 4. Margin analysis
    margin_analysis = compute_margin_analysis(results_df)
    if margin_analysis:
        with open(output_dir / "margin_analysis.json", 'w') as f:
            json.dump(margin_analysis, f, indent=2, default=str)
    
    # 5. Alignment metrics
    alignment_metrics = compute_alignment_metrics(results_df)
    if alignment_metrics:
        with open(output_dir / "alignment_metrics.json", 'w') as f:
            json.dump(alignment_metrics, f, indent=2, default=str)
    
    # 6. Potential comparison
    potential_comparison = compare_potentials(results_df)
    if potential_comparison:
        potential_df = pd.concat([df for df in potential_comparison.values()], 
                               keys=potential_comparison.keys(), axis=1)
        potential_df.to_csv(output_dir / "potential_comparison.csv")
    
    # 7. Depth comparison
    depth_comparison = compare_depths(results_df)
    if depth_comparison:
        depth_df = pd.concat([df for df in depth_comparison.values()], 
                           keys=depth_comparison.keys(), axis=1)
        depth_df.to_csv(output_dir / "depth_comparison.csv")
    
    # 8. Generate LaTeX tables
    try:
        generate_summary_table(results_df, output_dir / "summary_table.tex", format='latex')
    except Exception as e:
        print(f"Failed to generate LaTeX table: {e}")
    
    print(f"Comprehensive report generated in {output_dir}")


def load_results_from_directory(base_dir: str, 
                               pattern: str = "**/results.json") -> pd.DataFrame:
    """
    Load all results from a directory tree.
    
    Args:
        base_dir: Base directory to search
        pattern: Glob pattern for result files
        
    Returns:
        DataFrame with all results
    """
    base_path = Path(base_dir)
    result_files = list(base_path.glob(pattern))
    
    if not result_files:
        print(f"No result files found in {base_dir} with pattern {pattern}")
        return pd.DataFrame()
    
    experiment_dirs = [f.parent for f in result_files]
    
    return load_experiment_results(experiment_dirs)


def main():
    """Main aggregation function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Aggregate experiment results')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save aggregated results')
    parser.add_argument('--pattern', type=str, default='**/results.json',
                       help='Pattern for finding result files')
    
    args = parser.parse_args()
    
    print(f"Loading results from {args.input_dir}")
    results_df = load_results_from_directory(args.input_dir, args.pattern)
    
    if results_df.empty:
        print("No results found to aggregate")
        return
    
    print(f"Loaded {len(results_df)} experiments")
    print(f"Columns: {list(results_df.columns)}")
    
    generate_comprehensive_report(results_df, args.output_dir)
    print("Aggregation completed successfully!")


if __name__ == '__main__':
    main()
