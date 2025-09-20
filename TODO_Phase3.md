🧪 Cursor Prompt — Phase 3 (Empirical Exploration & Analysis)
You are an expert ML research engineer. Extend the repository to Phase 3: systematic experimentation, result aggregation, and automated figure/table generation. The goal is to produce publishable empirical results on the implicit bias of Mirror Descent in deep linear networks.
Do not modify core math (models, optimizer, potentials). Instead, enhance the framework for running, analyzing, and presenting experiments.
1) Grid Execution
Ensure run_grid.py supports:
Parallel execution across CPU cores (multiprocessing or joblib).
Resume/restart logic (skip runs if results already exist).
Progress logging with ETA.
Add flag --tag to label a grid run (stored in output directory).
2) Result Aggregation Enhancements
Extend aggregate.py to:
Group results by {potential, p, depth, dataset, digits, pca}.
Compute statistical summaries: mean, std, min, max across seeds.
Output master CSV + LaTeX tables directly in analysis_results/.
Provide per-condition JSONs for reproducibility.
3) Automated Plotting for Paper Figures
Extend plotting.py to generate publication-ready figures automatically after a grid run:
Angles vs epochs (deep vs linear MD, deep vs SVM/logreg).
Margins vs epochs.
Norm balance & alignment heatmaps across potentials/depths.
Boxplots of final angles/margins across seeds.
Decision boundary plots (for 2D synthetic, with baselines drawn).
Save figures in both .png and .pdf for paper inclusion.
Apply consistent formatting: serif fonts, gridlines, color palette.
4) Analysis Scripts
Add analysis/run_analysis.py which:
Loads a grid output directory.
Runs aggregate.py and plotting.py.
Produces a folder analysis_results/ containing:
Master CSV/LaTeX tables.
All plots (angles, margins, heatmaps).
An auto-generated Markdown report summarizing the results with inline figures.
5) Experiment Grids
Create configs for three levels:
quick_debug.yaml → small grid (1 seed, short epochs).
synthetic_potentials.yaml → potentials × depths × p-values × seeds.
mnist_pairs.yaml → {0 vs 1, 3 vs 5}, PCA={50,200}, potentials={quadratic, lp p=1.5, lp p=3}, depths={1,2}, 3 seeds.
Ensure configs are documented in README.
6) README Updates
Add a Phase 3 workflow example:
# Run synthetic grid
python -m runs.scripts.run_grid --config runs/configs/synthetic_potentials.yaml --tag synth_bias

# Aggregate + plot
python -m analysis.run_analysis --input_dir runs_out/20250319_synth_bias --output_dir analysis_results/synth_bias
7) Important Instruction
⚠️ Do not actually run the grids or analysis now. Only implement the framework and scripts so I can execute them later.
✅ With this Phase 3 upgrade, I’ll be able to:
Launch grids across potentials/depths/datasets.
Collect all results in one place.
Auto-generate figures, tables, and reports suitable for a Results section in a research paper.