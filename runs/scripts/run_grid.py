"""
Grid Experiment Runner for Mirror Descent Experiments.

This script runs systematic experiments by iterating through parameter combinations
specified in a YAML configuration file and collects results into a master CSV.
"""

import argparse
import itertools
import subprocess
import sys
import time
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import joblib

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


def check_experiment_exists(params: Dict[str, Any], output_base: str, tag: str = None) -> Optional[str]:
    """
    Check if experiment already exists and return its directory path.
    
    Args:
        params: Parameter dictionary
        output_base: Base directory for outputs
        tag: Optional tag for experiment
        
    Returns:
        Path to existing experiment directory if found, None otherwise
    """
    # Create a unique identifier for this parameter combination
    param_id = f"{params.get('potential', 'quadratic')}_L{params.get('depth', 1)}_p{params.get('p', 2.0)}_s{params.get('seed', 0)}"
    if params.get('dataset') == 'mnist':
        digits = params.get('digits', [0, 1])
        param_id += f"_d{digits[0]}{digits[1]}"
        if params.get('pca'):
            param_id += f"_pca{params['pca']}"
    
    # Search for existing experiment directories
    base_path = Path(output_base)
    if tag:
        search_pattern = f"*{tag}*{param_id}*"
    else:
        search_pattern = f"*{param_id}*"
    
    matching_dirs = list(base_path.glob(search_pattern))
    
    for exp_dir in matching_dirs:
        results_file = exp_dir / "results.json"
        if results_file.exists():
            return str(exp_dir)
    
    return None


def run_single_experiment(params: Dict[str, Any], output_base: str, tag: str = None, 
                         skip_existing: bool = True) -> Tuple[bool, str]:
    """
    Run a single experiment with given parameters.
    
    Args:
        params: Parameter dictionary
        output_base: Base directory for outputs
        tag: Optional tag for experiment
        skip_existing: Whether to skip if experiment already exists
        
    Returns:
        Tuple of (success, experiment_directory_path)
    """
    # Check if experiment already exists
    if skip_existing:
        existing_dir = check_experiment_exists(params, output_base, tag)
        if existing_dir:
            return True, existing_dir
    
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
    
    # Run experiment
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        exp_dir = result.stdout.strip().split('\n')[-1].split(': ')[-1]
        return True, exp_dir
    except subprocess.CalledProcessError as e:
        print(f"Experiment failed with error: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False, ""


def run_experiment_worker(args: Tuple) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Worker function for parallel experiment execution.
    
    Args:
        args: Tuple of (params, output_base, tag, skip_existing)
        
    Returns:
        Tuple of (success, experiment_directory_path, params)
    """
    params, output_base, tag, skip_existing = args
    success, exp_dir = run_single_experiment(params, output_base, tag, skip_existing)
    return success, exp_dir, params


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
                       help='Maximum number of parallel experiments')
    parser.add_argument('--skip_existing', action='store_true', default=True,
                       help='Skip experiments that already exist')
    parser.add_argument('--force_rerun', action='store_true', default=False,
                       help='Force rerun even if experiments exist')
    
    args = parser.parse_args()
    
    # Set skip_existing based on force_rerun
    skip_existing = args.skip_existing and not args.force_rerun
    
    print(f"Loading grid configuration from {args.config}")
    config = parse_grid_config(args.config)
    
    print("Generating parameter combinations...")
    combinations = generate_parameter_combinations(config)
    
    print(f"Total experiments to run: {len(combinations)}")
    print(f"Parallel execution: {args.max_parallel} workers")
    print(f"Skip existing experiments: {skip_existing}")
    
    # Create master experiment directory
    master_exp_dir = make_experiment_dir(base=args.output_base, tag=args.tag or "grid")
    
    # Save grid configuration
    save_json({
        'config': config,
        'combinations': combinations,
        'total_experiments': len(combinations),
        'parallel_workers': args.max_parallel,
        'skip_existing': skip_existing
    }, Path(master_exp_dir) / "grid_config.json")
    
    # Prepare arguments for parallel execution
    worker_args = []
    for i, params in enumerate(combinations):
        exp_tag = f"{args.tag}_exp_{i+1}" if args.tag else f"exp_{i+1}"
        worker_args.append((params, args.output_base, exp_tag, skip_existing))
    
    # Run experiments with progress tracking
    experiment_dirs = []
    successful_experiments = 0
    failed_experiments = 0
    skipped_experiments = 0
    
    start_time = time.time()
    
    if args.max_parallel > 1:
        # Parallel execution
        print(f"Starting parallel execution with {args.max_parallel} workers...")
        
        with ProcessPoolExecutor(max_workers=args.max_parallel) as executor:
            # Submit all jobs
            future_to_params = {executor.submit(run_experiment_worker, args): args for args in worker_args}
            
            # Process completed jobs with progress bar
            with tqdm(total=len(combinations), desc="Running experiments") as pbar:
                for future in as_completed(future_to_params):
                    success, exp_dir, params = future.result()
                    
                    if success:
                        experiment_dirs.append(exp_dir)
                        if exp_dir and Path(exp_dir).exists():
                            results_file = Path(exp_dir) / "results.json"
                            if results_file.exists():
                                successful_experiments += 1
                            else:
                                skipped_experiments += 1
                        else:
                            skipped_experiments += 1
                    else:
                        failed_experiments += 1
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'Success': successful_experiments,
                        'Failed': failed_experiments,
                        'Skipped': skipped_experiments
                    })
    else:
        # Sequential execution
        print("Starting sequential execution...")
        
        for i, (params, output_base, tag, skip_existing) in enumerate(tqdm(worker_args, desc="Running experiments")):
            print(f"\n--- Running experiment {i+1}/{len(combinations)} ---")
            print(f"Parameters: {params}")
            
            success, exp_dir = run_single_experiment(params, output_base, tag, skip_existing)
            
            if success:
                experiment_dirs.append(exp_dir)
                if exp_dir and Path(exp_dir).exists():
                    results_file = Path(exp_dir) / "results.json"
                    if results_file.exists():
                        successful_experiments += 1
                    else:
                        skipped_experiments += 1
                else:
                    skipped_experiments += 1
            else:
                failed_experiments += 1
    
    elapsed_time = time.time() - start_time
    
    print(f"\n=== Grid Execution Summary ===")
    print(f"Total experiments: {len(combinations)}")
    print(f"Successful: {successful_experiments}")
    print(f"Failed: {failed_experiments}")
    print(f"Skipped (already existed): {skipped_experiments}")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Average time per experiment: {elapsed_time/len(combinations):.2f} seconds")
    
    # Collect results
    if experiment_dirs:
        print("\nCollecting results...")
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
    
    # Save execution summary
    execution_summary = {
        'total_experiments': len(combinations),
        'successful_experiments': successful_experiments,
        'failed_experiments': failed_experiments,
        'skipped_experiments': skipped_experiments,
        'execution_time_seconds': elapsed_time,
        'parallel_workers': args.max_parallel,
        'skip_existing': skip_existing
    }
    
    save_json(execution_summary, Path(master_exp_dir) / "execution_summary.json")
    
    print(f"\nGrid experiment batch completed. Master directory: {master_exp_dir}")


if __name__ == '__main__':
    main()
