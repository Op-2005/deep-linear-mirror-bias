You are an expert ML research engineer. Upgrade the repository from Phase 1 (complete MD/DLN implementation) into a research-grade experimental framework for Phase 2: running systematic experiments, logging results, and generating analysis plots. Do not change the core model/optimizer/potential code; instead, extend the repo for experiment orchestration and visualization.
1) Experiment Management
Add a utility module utils/experiment.py with functions:
make_experiment_dir(base="runs_out", tag=None): create timestamped directory with subfolders figs/, tables/, logs/.
save_json(obj, path), save_yaml(obj, path).
set_seeds(seed) for torch, numpy, sklearn.
Modify train_synth.py to:
Auto-create an experiment directory for each run.
Save CLI args + config as config.yaml.
Save metrics (angles, margins, alignment, norm_balance) over epochs into a .jsonl or .csv.
Save decision boundary plot (if 2D) into figs/.
2) Batch Run Support
Add runs/scripts/run_grid.py that:
Reads a YAML config specifying a grid of parameters (e.g. potentials, depths, seeds).
Iterates through combinations and calls train_synth.py via subprocess or function call.
Collects final JSON metrics into a master CSV for comparison.
Example grid config:
dataset: synthetic
input_dim: 2
n_samples: 512
separation: 3.0
noise: 0.2
epochs: 1000
lr: 0.05
potentials: [quadratic, lp]
p_values: [1.5, 3.0]
depths: [1, 2]
seeds: [0, 1, 2]
3) Plotting & Visualization
Add eval/plotting.py with functions:
plot_angles_over_time(results, savepath): line plots (mean ± std across seeds) of angle vs epochs.
plot_margins_over_time(results, savepath).
plot_norm_balance(results, savepath).
plot_decision_boundary(model, X, y, savepath) for 2D synthetic.
Support both per-run plots and aggregated plots across seeds/potentials.
4) Result Aggregation
Add eval/aggregate.py that:
Reads multiple JSON/CSV results from experiment dirs.
Groups by potential/depth/p.
Outputs summary tables:
Final angles (deep vs SVM/logreg, deep vs linear MD).
Final margins.
Norm balance and alignment metrics.
Saves summary as CSV + optional LaTeX table.
5) MNIST Integration
Extend train_synth.py to support --dataset mnist --digits 0 1 or --digits 3 5.
For MNIST runs: skip decision boundary plot, but still log metrics and save per-epoch CSV.
Add CLI flag --pca N to project MNIST to N dimensions before training.
6) Command Examples
Update README with Phase 2 example commands:
# Run single synthetic experiment
python -m runs.scripts.train_synth --potential quadratic --L 2 --input_dim 2 \
  --n_samples 512 --separation 3.0 --noise 0.2 --lr 0.1 --epochs 1000 --seed 0

# Run grid of synthetic experiments
python -m runs.scripts/run_grid --config runs/configs/synth_grid.yaml

# Run MNIST 0 vs 1 with lp potential
python -m runs.scripts/train_synth --dataset mnist --digits 0 1 \
  --potential lp --p 1.5 --L 2 --lr 0.02 --epochs 20 --pca 50 --seed 0
7) Analysis Workflow
Ensure outputs are reproducible and organized:
Each run → its own directory with config, logs, figures, metrics.
Grids → master CSV with all final metrics.
Provide plotting utilities to quickly generate:
Angles vs epochs (deep vs linear MD vs baselines).
Margins vs epochs.
Alignment & norm balance heatmaps.
This should allow fast paper-ready figures.
8) Important Instruction
⚠️ Do not actually execute training or experiments yet. Your task is only to implement and update the code to support Phase 2 (logging, grids, plotting, aggregation). I will run experiments later.
✅ With this upgrade, the repo will be ready for systematic empirical exploration: you’ll be able to run grids, log everything reproducibly, and generate analysis figures/tables for synthetic and MNIST experiments.