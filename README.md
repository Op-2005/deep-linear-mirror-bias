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

### Phase 3: Systematic Experimentation & Analysis Framework

This repository now supports comprehensive systematic experimentation with parallel execution, automated analysis, and publication-ready figure generation. The framework provides complete pipelines from experiment execution to paper-ready results.

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

### Phase 3: Complete Analysis Workflow

**Step 1: Run Systematic Experiments**
```bash
# Run synthetic potential comparison (192 experiments)
python -m runs.scripts.run_grid --config runs/configs/synthetic_potentials.yaml \
  --tag synth_bias --max_parallel 4

# Run MNIST digit pair experiments (96 experiments)  
python -m runs.scripts.run_grid --config runs/configs/mnist_pairs.yaml \
  --tag mnist_bias --max_parallel 4

# Quick debug test (4 experiments)
python -m runs.scripts.run_grid --config runs/configs/quick_debug.yaml \
  --tag debug_test
```

**Step 2: Automated Analysis Pipeline**
```bash
# Complete analysis: aggregation + plots + report
python -m analysis.run_analysis --input_dir runs_out/20250319_synth_bias \
  --output_dir analysis_results/synth_bias

# MNIST analysis
python -m analysis.run_analysis --input_dir runs_out/20250319_mnist_bias \
  --output_dir analysis_results/mnist_bias
```

### Advanced Features

**Parallel Execution with Progress Tracking:**
```bash
# Run with 8 parallel workers and progress bars
python -m runs.scripts.run_grid --config runs/configs/synthetic_potentials.yaml \
  --max_parallel 8 --tag parallel_test
```

**Resume/Restart Logic:**
```bash
# Skip existing experiments (default behavior)
python -m runs.scripts.run_grid --config runs/configs/synthetic_potentials.yaml \
  --skip_existing

# Force rerun all experiments
python -m runs.scripts.run_grid --config runs/configs/synthetic_potentials.yaml \
  --force_rerun
```

**Custom Analysis:**
```bash
# Skip plots for faster analysis
python -m analysis.run_analysis --input_dir runs_out/20250319_synth_bias \
  --output_dir analysis_results/synth_bias --skip_plots

# Skip LaTeX tables
python -m analysis.run_analysis --input_dir runs_out/20250319_synth_bias \
  --output_dir analysis_results/synth_bias --skip_tables
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

**Individual Experiments:**
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

**Grid Experiments:**
```
runs_out/20241201_143022_grid_tag/
├── grid_config.json         # Grid specification
├── execution_summary.json   # Execution statistics
├── tables/master_results.csv    # All experiments in one table
└── [individual experiment directories]
```

**Analysis Results:**
```
analysis_results/experiment_tag/
├── analysis_report.md           # Automated Markdown report
├── analysis_metadata.json       # Analysis metadata
├── figures/                     # Publication-ready plots (PNG + PDF)
│   ├── margins_by_parameters.png
│   ├── heatmaps.png
│   ├── margin_distributions.png
│   └── [other figures]
├── tables/                      # CSV data tables
│   ├── aggregated_results.csv
│   ├── angle_analysis.csv
│   └── [other tables]
├── latex_tables/                # LaTeX tables for papers
│   ├── main_results.tex
│   ├── significance_tests.tex
│   └── [other tables]
└── per_condition_results/       # JSON files per condition
    ├── quadratic_depth_1.json
    └── [other conditions]
```

## Citation

*TODO: Add citation information*
