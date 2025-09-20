# Implicit Bias of Mirror Descent in Deep Linear Networks

This project explores the implicit bias of Mirror Descent (MD) in deep linear networks, comparing linear vs. deep classifiers under different potentials. We investigate how different potential functions in Mirror Descent optimization affect the convergence behavior and generalization properties of learned models.

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd deep-linear-mirror-bias
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Repository Structure

```
deep-linear-mirror-bias/
├── core/                    # Core model and optimizer implementations
│   ├── models.py           # DeepLinear, Linear models
│   ├── md_optimizer.py     # Mirror Descent optimizer
│   └── potentials.py       # Quadratic, Lp, scaled potentials
├── data/                   # Data generation and loading
│   ├── synthetic.py        # Gaussian data generators
│   └── mnist.py           # MNIST binary loader
├── eval/                   # Evaluation metrics and baselines
│   ├── metrics.py         # Margins, angles, alignment
│   └── baselines.py       # SVM, p-margin solvers
├── runs/                   # Training and evaluation scripts
│   ├── configs/           # YAML configuration files
│   └── scripts/           # Training and evaluation scripts
├── reports/               # Results and visualizations
│   ├── figs/             # Plots and figures
│   └── tables/           # CSV/TeX tables
├── paper/                # LaTeX paper
│   └── main.tex         # Main paper document
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Experiments

### Phase 2: Research-Grade Experimental Framework

This repository now supports systematic experiments with automatic logging, result aggregation, and visualization. Each experiment is automatically organized in timestamped directories with comprehensive metrics tracking.

### Single Experiment Examples

**Synthetic Data:**
```bash
# Run single synthetic experiment
python -m runs.scripts.train_synth --potential quadratic --L 2 --input_dim 2 \
  --n_samples 512 --separation 3.0 --noise 0.2 --lr 0.1 --epochs 1000 --seed 0
```

**MNIST Binary Classification:**
```bash
# Run MNIST 0 vs 1 with lp potential
python -m runs.scripts.train_synth --dataset mnist --digits 0 1 \
  --potential lp --p 1.5 --L 2 --lr 0.02 --epochs 50 --pca 50 --seed 0

# Run MNIST 3 vs 5 with quadratic potential
python -m runs.scripts.train_synth --dataset mnist --digits 3 5 \
  --potential quadratic --L 3 --lr 0.01 --epochs 30 --seed 0
```

### Grid Experiments

**Synthetic Grid:**
```bash
# Run systematic synthetic experiments
python -m runs.scripts.run_grid --config runs/configs/synth_grid.yaml

# Quick test with minimal parameters
python -m runs.scripts.run_grid --config runs/configs/quick_test.yaml
```

**MNIST Grid:**
```bash
# Run systematic MNIST experiments
python -m runs.scripts.run_grid --config runs/configs/mnist_grid.yaml
```

### Result Aggregation and Analysis

**Aggregate Results:**
```bash
# Aggregate results from experiment directory
python -m eval.aggregate --input_dir runs_out/20241201_143022_grid \
  --output_dir analysis_results
```

**Generate Publication Plots:**
```python
from eval.plotting import create_publication_plots
import pandas as pd

# Load aggregated results
results_df = pd.read_csv('analysis_results/aggregated_results.csv')

# Create publication-ready plots
create_publication_plots(results_df, 'publication_figures/')
```

### Available Potentials

- `quadratic`: Standard L2 potential (gradient descent)
- `lp`: Lp potential with configurable p (use `--p` argument)
- `layer_scaled`: Layer-scaled quadratic potential
- `scaled`: Magnitude-dependent scaled potential

### Key Metrics Tracked

- **Margin**: Minimum distance to decision boundary
- **Angles**: Alignment with SVM and logistic regression baselines
- **Layer Alignment**: Cosine similarity between layer singular vectors
- **Norm Balance**: Ratio of max/min Frobenius norms across layers
- **Training Curves**: Loss, accuracy, margin evolution over epochs
- **Spectral Properties**: Condition numbers and singular values

### Output Organization

Each experiment creates a timestamped directory with:
```
runs_out/20241201_143022_experiment_name/
├── config.yaml              # Experiment configuration
├── results.json             # Final metrics
├── experiment_summary.json  # Experiment overview
├── figs/                    # Plots and visualizations
│   └── decision_boundary.png
├── tables/                  # Data tables
│   └── metrics_history.csv
└── logs/                    # Training logs
    └── metrics_history.jsonl
```

Grid experiments additionally create:
- `master_results.csv`: All results in one table
- `grid_config.json`: Grid configuration used

## Citation

*TODO: Add citation information*
