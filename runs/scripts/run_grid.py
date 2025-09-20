"""
Grid Experiment Runner for Mirror Descent Experiments.

This script runs systematic experiments by iterating through parameter combinations
specified in a YAML configuration file and collects results into a master CSV.
"""

import argparse
import itertools
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from utils.experiment import load_yaml, save_json, make_experiment_dir


def parse_grid_config(config_path: str) -> Dict[str, Any]:
    """
    Parse grid configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Parsed configuration dictionary
    """
    config = load_yaml(config_path)
    
    # Validate required fields
    required_fields = ['dataset', 'epochs', 'lr']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Required field '{field}' missing from config")
    
    return config


def generate_parameter_combinations(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate all parameter combinations from grid configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of parameter combinations
    """
    # Extract parameter lists
    param_lists = {}
    
    # Dataset parameters (single values)
    single_params = ['dataset', 'input_dim', 'n_samples', 'separation', 'noise', 
                     'epochs', 'lr', 'log_frequency']
    
    for param in single_params:
        if param in config:
            param_lists[param] = [config[param]]
    
    # Grid parameters (lists)
    grid_params = ['potentials', 'p_values', 'depths', 'seeds']
    
    for param in grid_params:
        if param in config:
            param_lists[param] = config[param]
    
    # MNIST-specific parameters
    if config.get('dataset') == 'mnist':
        if 'digits_combinations' in config:
            param_lists['digits'] = config['digits_combinations']
        if 'pca_components' in config:
            param_lists['pca'] = config['pca_components']
    
    # Generate all combinations
    param_names = list(param_lists.keys())
    param_values = list(param_lists.values())
    
    combinations = []
    for combination in itertools.product(*param_values):
        param_dict = dict(zip(param_names, combination))
        combinations.append(param_dict)
    
    return combinations


def run_single_experiment(params: Dict[str, Any], output_base: str, tag: str = None) -> str:
    """
    Run a single experiment with given parameters.
    
    Args:
        params: Parameter dictionary
        output_base: Base directory for outputs
        tag: Optional tag for experiment
        
    Returns:
        Path to experiment directory
    """
    # Build command
    cmd = [sys.executable, '-m', 'runs.scripts.train_synth']
    
    # Add parameters to command
    for key, value in params.items():
        if key == 'potentials':
            cmd.extend(['--potential', str(value)])
        elif key == 'p_values':
            cmd.extend(['--p', str(value)])
        elif key == 'depths':
            cmd.extend(['--L', str(value)])
        elif key == 'digits':
            cmd.extend(['--digits'] + [str(d) for d in value])
        elif key in ['pca']:
            if value is not None:
                cmd.extend([f'--{key}', str(value)])
        else:
            # Map parameter names to CLI arguments
            arg_map = {
                'input_dim': '--input_dim',
                'n_samples': '--n_samples', 
                'separation': '--separation',
                'noise': '--noise',
                'epochs': '--epochs',
                'lr': '--lr',
                'log_frequency': '--log_frequency',
                'seeds': '--seed'
            }
            
            if key in arg_map:
                cmd.extend([arg_map[key], str(value)])
    
    # Add output directory
    cmd.extend(['--output_base', output_base])
    if tag:
        cmd.extend(['--tag', tag])
    
    print(f"Running: {' '.join(cmd)}")
    
    # Run experiment
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Experiment completed successfully")
        return result.stdout.strip().split('\n')[-1].split(': ')[-1]  # Extract exp dir from output
    except subprocess.CalledProcessError as e:
        print(f"Experiment failed with error: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise


def collect_results(experiment_dirs: List[str]) -> pd.DataFrame:
    """
    Collect results from multiple experiment directories.
    
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
                from utils.experiment import load_json
                result_data = load_json(results_file)
                results.append(result_data)
            except Exception as e:
                print(f"Failed to load results from {results_file}: {e}")
        else:
            print(f"Results file not found: {results_file}")
    
    return pd.DataFrame(results)


def main():
    """Main grid runner function."""
    parser = argparse.ArgumentParser(description='Run grid experiments for Mirror Descent')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML configuration file')
    parser.add_argument('--output_base', type=str, default='runs_out',
                       help='Base directory for experiment outputs')
    parser.add_argument('--tag', type=str, default=None,
                       help='Tag for experiment batch')
    parser.add_argument('--max_parallel', type=int, default=1,
                       help='Maximum number of parallel experiments (currently unused)')
    
    args = parser.parse_args()
    
    print(f"Loading grid configuration from {args.config}")
    config = parse_grid_config(args.config)
    
    print("Generating parameter combinations...")
    combinations = generate_parameter_combinations(config)
    
    print(f"Total experiments to run: {len(combinations)}")
    
    # Create master experiment directory
    master_exp_dir = make_experiment_dir(base=args.output_base, tag=args.tag or "grid")
    
    # Save grid configuration
    save_json({
        'config': config,
        'combinations': combinations,
        'total_experiments': len(combinations)
    }, Path(master_exp_dir) / "grid_config.json")
    
    # Run experiments
    experiment_dirs = []
    successful_experiments = 0
    
    for i, params in enumerate(combinations):
        print(f"\n--- Running experiment {i+1}/{len(combinations)} ---")
        print(f"Parameters: {params}")
        
        try:
            exp_dir = run_single_experiment(params, args.output_base, f"{args.tag}_exp_{i+1}" if args.tag else f"exp_{i+1}")
            experiment_dirs.append(exp_dir)
            successful_experiments += 1
        except Exception as e:
            print(f"Experiment {i+1} failed: {e}")
            continue
    
    print(f"\nCompleted {successful_experiments}/{len(combinations)} experiments successfully")
    
    # Collect results
    if experiment_dirs:
        print("Collecting results...")
        results_df = collect_results(experiment_dirs)
        
        # Save master results
        master_results_path = Path(master_exp_dir) / "tables" / "master_results.csv"
        results_df.to_csv(master_results_path, index=False)
        
        print(f"Master results saved to {master_results_path}")
        print(f"Results shape: {results_df.shape}")
        
        # Print summary statistics
        if not results_df.empty:
            print("\nSummary statistics:")
            numeric_cols = results_df.select_dtypes(include=['number']).columns
            summary = results_df[numeric_cols].describe()
            print(summary)
    else:
        print("No successful experiments to collect results from")
    
    print(f"\nGrid experiment batch completed. Master directory: {master_exp_dir}")


if __name__ == '__main__':
    main()
