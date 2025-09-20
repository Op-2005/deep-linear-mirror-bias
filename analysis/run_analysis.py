"""
Complete Analysis Pipeline for Mirror Descent Experiments.

This script provides a comprehensive analysis pipeline that loads grid experiment results,
runs aggregation, generates plots, and creates an automated Markdown report.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from eval.aggregate import (
    load_results_from_directory, 
    generate_comprehensive_report,
    generate_latex_tables,
    aggregate_by_parameters
)
from eval.plotting import create_publication_plots


def generate_markdown_report(results_df: pd.DataFrame, output_dir: str, 
                           input_dir: str) -> None:
    """
    Generate an automated Markdown report summarizing the results.
    
    Args:
        results_df: DataFrame with experiment results
        output_dir: Directory to save the report
        input_dir: Input directory path for reference
    """
    output_dir = Path(output_dir)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Start building the report
    report_lines = [
        "# Mirror Descent Implicit Bias Analysis Report",
        "",
        f"**Generated on:** {timestamp}",
        f"**Input Directory:** `{input_dir}`",
        f"**Analysis Directory:** `{output_dir}`",
        "",
        "## Executive Summary",
        "",
        f"This report summarizes the results of {len(results_df)} Mirror Descent experiments.",
        ""
    ]
    
    # Add dataset information
    if 'dataset' in results_df.columns:
        datasets = results_df['dataset'].unique()
        report_lines.extend([
            "### Datasets",
            f"- **Datasets analyzed:** {', '.join(datasets)}",
            ""
        ])
        
        for dataset in datasets:
            count = len(results_df[results_df['dataset'] == dataset])
            report_lines.append(f"- **{dataset}:** {count} experiments")
        report_lines.append("")
    
    # Add potential function information
    if 'potential' in results_df.columns:
        potentials = results_df['potential'].unique()
        report_lines.extend([
            "### Potential Functions",
            f"- **Potential functions tested:** {', '.join(potentials)}",
            ""
        ])
        
        for potential in potentials:
            count = len(results_df[results_df['potential'] == potential])
            report_lines.append(f"- **{potential}:** {count} experiments")
        report_lines.append("")
    
    # Add network depth information
    if 'depth' in results_df.columns:
        depths = sorted(results_df['depth'].unique())
        report_lines.extend([
            "### Network Depths",
            f"- **Network depths tested:** {', '.join(map(str, depths))}",
            ""
        ])
    
    # Add key findings
    report_lines.extend([
        "## Key Findings",
        ""
    ])
    
    # Margin analysis
    if 'final_margin' in results_df.columns:
        margin_stats = results_df['final_margin'].describe()
        best_margin = results_df.loc[results_df['final_margin'].idxmax()]
        worst_margin = results_df.loc[results_df['final_margin'].idxmin()]
        
        report_lines.extend([
            "### Margin Analysis",
            f"- **Mean final margin:** {margin_stats['mean']:.4f} ± {margin_stats['std']:.4f}",
            f"- **Best margin:** {best_margin['final_margin']:.4f} (Potential: {best_margin['potential']}, Depth: {best_margin['depth']})",
            f"- **Worst margin:** {worst_margin['final_margin']:.4f} (Potential: {worst_margin['potential']}, Depth: {worst_margin['depth']})",
            ""
        ])
    
    # Angle analysis
    if 'angle_to_svm' in results_df.columns:
        angle_stats = results_df['angle_to_svm'].describe()
        best_alignment = results_df.loc[results_df['angle_to_svm'].idxmin()]  # Lower angle = better alignment
        
        report_lines.extend([
            "### Alignment with SVM Baseline",
            f"- **Mean angle to SVM:** {angle_stats['mean']:.2f}° ± {angle_stats['std']:.2f}°",
            f"- **Best alignment:** {best_alignment['angle_to_svm']:.2f}° (Potential: {best_alignment['potential']}, Depth: {best_alignment['depth']})",
            ""
        ])
    
    # Norm balance analysis
    if 'norm_balance' in results_df.columns:
        norm_stats = results_df['norm_balance'].describe()
        best_balance = results_df.loc[results_df['norm_balance'].idxmin()]  # Lower = better balance
        
        report_lines.extend([
            "### Norm Balance Analysis",
            f"- **Mean norm balance:** {norm_stats['mean']:.2f} ± {norm_stats['std']:.2f}",
            f"- **Best balance:** {best_balance['norm_balance']:.2f} (Potential: {best_balance['potential']}, Depth: {best_balance['depth']})",
            ""
        ])
    
    # Add figures section
    report_lines.extend([
        "## Figures",
        "",
        "The following figures provide detailed visualizations of the experimental results:",
        ""
    ])
    
    # List available figures
    figures_dir = output_dir / "figures"
    if figures_dir.exists():
        figure_files = list(figures_dir.glob("*.png"))
        
        for fig_file in sorted(figure_files):
            fig_name = fig_file.stem.replace('_', ' ').title()
            report_lines.extend([
                f"### {fig_name}",
                f"",
                f"![{fig_name}]({fig_file.name})",
                ""
            ])
    
    # Add tables section
    report_lines.extend([
        "## Tables",
        "",
        "Summary tables and statistical analyses are available in the following formats:",
        "",
        "- **CSV files:** `tables/` directory contains machine-readable data",
        "- **LaTeX files:** `latex_tables/` directory contains publication-ready tables",
        "- **JSON files:** `per_condition_results/` directory contains detailed per-condition data",
        ""
    ])
    
    # Add methodology section
    report_lines.extend([
        "## Methodology",
        "",
        "### Experimental Setup",
        "- Deep linear networks with Mirror Descent optimization",
        "- Multiple potential functions tested",
        "- Systematic grid search over parameters",
        "- Statistical analysis across multiple random seeds",
        "",
        "### Metrics Computed",
        "- **Margin:** Minimum distance to decision boundary",
        "- **Angles:** Alignment with SVM and logistic regression baselines", 
        "- **Layer Alignment:** Cosine similarity between layer singular vectors",
        "- **Norm Balance:** Ratio of max/min Frobenius norms across layers",
        "- **Training/Validation Accuracy:** Classification performance",
        ""
    ])
    
    # Add reproducibility section
    report_lines.extend([
        "## Reproducibility",
        "",
        "All experimental configurations and results are preserved for reproducibility:",
        "",
        "- **Grid configuration:** `grid_config.json`",
        "- **Execution summary:** `execution_summary.json`",
        "- **Individual results:** `results.json` files in each experiment directory",
        "- **Metrics history:** `metrics_history.jsonl` files for time series analysis",
        ""
    ])
    
    # Add conclusion
    report_lines.extend([
        "## Conclusion",
        "",
        "This analysis provides comprehensive insights into the implicit bias properties of Mirror Descent optimization in deep linear networks. The results demonstrate how different potential functions affect the convergence behavior and generalization properties of the learned models.",
        "",
        "For detailed statistical analyses and publication-ready figures, refer to the accompanying files in this analysis directory.",
        ""
    ])
    
    # Write the report
    report_path = output_dir / "analysis_report.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Markdown report saved to {report_path}")


def main():
    """Main analysis pipeline function."""
    parser = argparse.ArgumentParser(description='Run comprehensive analysis on Mirror Descent experiment results')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing grid experiment results')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save analysis results')
    parser.add_argument('--pattern', type=str, default='**/results.json',
                       help='Pattern for finding result files')
    parser.add_argument('--skip_plots', action='store_true', default=False,
                       help='Skip figure generation (faster for testing)')
    parser.add_argument('--skip_tables', action='store_true', default=False,
                       help='Skip LaTeX table generation')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    print(f"Loading results from {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    results_df = load_results_from_directory(args.input_dir, args.pattern)
    
    if results_df.empty:
        print("No results found to analyze")
        return
    
    print(f"Loaded {len(results_df)} experiments")
    print(f"Columns: {list(results_df.columns)}")
    
    # Run comprehensive aggregation
    print("\n=== Running Aggregation ===")
    generate_comprehensive_report(results_df, str(output_dir))
    
    # Generate LaTeX tables
    if not args.skip_tables:
        print("\n=== Generating LaTeX Tables ===")
        latex_dir = output_dir / "latex_tables"
        generate_latex_tables(results_df, str(latex_dir))
    
    # Generate publication plots
    if not args.skip_plots:
        print("\n=== Generating Publication Plots ===")
        figures_dir = output_dir / "figures"
        create_publication_plots(results_df, str(figures_dir))
    
    # Generate Markdown report
    print("\n=== Generating Markdown Report ===")
    generate_markdown_report(results_df, str(output_dir), str(input_dir))
    
    # Save analysis metadata
    metadata = {
        'analysis_timestamp': datetime.now().isoformat(),
        'input_directory': str(input_dir),
        'output_directory': str(output_dir),
        'total_experiments': len(results_df),
        'columns': list(results_df.columns),
        'datasets': list(results_df['dataset'].unique()) if 'dataset' in results_df.columns else [],
        'potentials': list(results_df['potential'].unique()) if 'potential' in results_df.columns else [],
        'depths': list(results_df['depth'].unique()) if 'depth' in results_df.columns else []
    }
    
    with open(output_dir / "analysis_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"\n=== Analysis Complete ===")
    print(f"Results saved to: {output_dir}")
    print(f"Markdown report: {output_dir / 'analysis_report.md'}")
    print(f"Figures: {output_dir / 'figures'}")
    print(f"Tables: {output_dir / 'tables'}")
    print(f"LaTeX tables: {output_dir / 'latex_tables'}")


if __name__ == '__main__':
    main()
