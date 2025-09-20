"""
Experiment Management Utilities.

This module provides utilities for managing experiments, creating directories,
saving results, and ensuring reproducibility in the Mirror Descent research framework.
"""

import os
import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union
import torch
import numpy as np
import sklearn


def make_experiment_dir(base: str = "runs_out", tag: Optional[str] = None) -> str:
    """
    Create a timestamped experiment directory with subfolders.
    
    Args:
        base: Base directory for experiments
        tag: Optional tag to add to directory name
        
    Returns:
        Path to the created experiment directory
    """
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directory name
    if tag:
        dir_name = f"{timestamp}_{tag}"
    else:
        dir_name = timestamp
    
    # Create full path
    exp_dir = Path(base) / dir_name
    
    # Create directory and subdirectories
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "figs").mkdir(exist_ok=True)
    (exp_dir / "tables").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    
    print(f"Created experiment directory: {exp_dir}")
    return str(exp_dir)


def save_json(obj: Any, path: Union[str, Path]) -> None:
    """
    Save object to JSON file with proper serialization.
    
    Args:
        obj: Object to save
        path: Path to save the JSON file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to serializable format
    def convert_to_serializable(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_obj = convert_to_serializable(obj)
    
    with open(path, 'w') as f:
        json.dump(serializable_obj, f, indent=2)


def save_yaml(obj: Any, path: Union[str, Path]) -> None:
    """
    Save object to YAML file.
    
    Args:
        obj: Object to save
        path: Path to save the YAML file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        yaml.dump(obj, f, default_flow_style=False, indent=2)


def set_seeds(seed: int) -> None:
    """
    Set random seeds for reproducibility across torch, numpy, and sklearn.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Set sklearn random state if available
    try:
        sklearn.utils.check_random_state(seed)
    except:
        pass
    
    # Set Python hash seed for deterministic behavior
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Set random seeds to {seed}")


def load_json(path: Union[str, Path]) -> Any:
    """
    Load object from JSON file.
    
    Args:
        path: Path to the JSON file
        
    Returns:
        Loaded object
    """
    with open(path, 'r') as f:
        return json.load(f)


def load_yaml(path: Union[str, Path]) -> Any:
    """
    Load object from YAML file.
    
    Args:
        path: Path to the YAML file
        
    Returns:
        Loaded object
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_experiment_config(config: Dict[str, Any], exp_dir: str) -> None:
    """
    Save experiment configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        exp_dir: Experiment directory path
    """
    config_path = Path(exp_dir) / "config.yaml"
    save_yaml(config, config_path)
    print(f"Saved configuration to {config_path}")


def save_metrics_history(metrics_history: list, exp_dir: str) -> None:
    """
    Save metrics history to JSONL file.
    
    Args:
        metrics_history: List of metric dictionaries
        exp_dir: Experiment directory path
    """
    metrics_path = Path(exp_dir) / "logs" / "metrics_history.jsonl"
    
    with open(metrics_path, 'w') as f:
        for metrics in metrics_history:
            json.dump(metrics, f)
            f.write('\n')
    
    print(f"Saved metrics history to {metrics_path}")


def save_metrics_csv(metrics_history: list, exp_dir: str) -> None:
    """
    Save metrics history to CSV file.
    
    Args:
        metrics_history: List of metric dictionaries
        exp_dir: Experiment directory path
    """
    import pandas as pd
    
    # Convert to DataFrame
    df = pd.DataFrame(metrics_history)
    
    # Save to CSV
    csv_path = Path(exp_dir) / "tables" / "metrics_history.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"Saved metrics history to {csv_path}")


def create_experiment_summary(exp_dir: str, final_metrics: Dict[str, Any]) -> None:
    """
    Create a summary of the experiment.
    
    Args:
        exp_dir: Experiment directory path
        final_metrics: Final metrics dictionary
    """
    summary = {
        'experiment_dir': exp_dir,
        'timestamp': datetime.now().isoformat(),
        'final_metrics': final_metrics
    }
    
    summary_path = Path(exp_dir) / "experiment_summary.json"
    save_json(summary, summary_path)
    print(f"Created experiment summary: {summary_path}")


def get_experiment_info(exp_dir: str) -> Dict[str, Any]:
    """
    Get information about an experiment from its directory.
    
    Args:
        exp_dir: Experiment directory path
        
    Returns:
        Dictionary with experiment information
    """
    exp_dir = Path(exp_dir)
    
    info = {
        'directory': str(exp_dir),
        'name': exp_dir.name,
        'created': datetime.fromtimestamp(exp_dir.stat().st_ctime).isoformat(),
        'files': []
    }
    
    # List all files in the experiment directory
    for file_path in exp_dir.rglob('*'):
        if file_path.is_file():
            info['files'].append({
                'path': str(file_path.relative_to(exp_dir)),
                'size': file_path.stat().st_size,
                'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            })
    
    return info


def cleanup_experiment(exp_dir: str, keep_logs: bool = True) -> None:
    """
    Clean up experiment directory by removing temporary files.
    
    Args:
        exp_dir: Experiment directory path
        keep_logs: Whether to keep log files
    """
    exp_dir = Path(exp_dir)
    
    # Files to remove
    temp_extensions = ['.tmp', '.temp', '.pyc', '__pycache__']
    
    for file_path in exp_dir.rglob('*'):
        if file_path.is_file():
            if any(file_path.name.endswith(ext) for ext in temp_extensions):
                file_path.unlink()
                print(f"Removed temporary file: {file_path}")
        elif file_path.is_dir() and file_path.name == '__pycache__':
            import shutil
            shutil.rmtree(file_path)
            print(f"Removed directory: {file_path}")
    
    print(f"Cleaned up experiment directory: {exp_dir}")
