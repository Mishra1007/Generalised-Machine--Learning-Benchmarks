Experiment Protocol

This protocol describes the exact steps used to prepare datasets, run model evaluations, compute metrics, and execute statistical tests so an independent researcher can reproduce experiments.

1. Dataset Preparation

- Dataset provenance: Record the source, version, and any filters applied. Place original raw data under `data/` (or note external source and download script).
- Split generation: Generate cross-validation splits deterministically using a seeded RNG. Save split definitions and mapping files to `artifacts/splits/<dataset>/`.

2. Preprocessing

- Apply deterministic, documented preprocessing steps via `preprocessing/pipelines.py`.
- Record preprocessing configuration in `configs/` and save per-run preprocessing artifacts (transforms, scalers).

3. Model Training & Evaluation

- Training: Use model wrappers in `models/` ensuring deterministic behavior where possible (seed control, stable optimizer settings).
- Per-fold evaluation: For each fold, compute primary and auxiliary metrics and write a row to `results/<dataset>/raw_results.csv` with columns including: `model`, `repetition_id`, `fold_id`, metric columns, and optional metadata.
- Aggregation: After all folds, run the normalization and CBS computation pipeline to produce `normalized_results.csv` and `cbs_scores.csv`.

4. Statistical Testing

- Alignment: Build the aligned score matrix by joining per-fold scores on (`repetition_id`, `fold_id`) and models present in the experiment.
- Global test: Run Friedman test across models.
- Post-hoc: Run Nemenyi test when Friedman is significant. For pairwise confirmation, run Wilcoxon Signed-Rank tests with Holm adjustment.
- Effect sizes & CIs: For each pair reported, include Cohen's d (paired), Cliff's Delta, and bootstrap or t-based CIs for mean differences.

5. Analysis Reproducibility

- Use the CLI scripts in `scripts/` to re-run analysis from stored artifacts without re-training models.
- All analysis parameters (random seeds, number of bootstrap iterations, MC iterations) are explicit arguments to analysis entrypoints and must be recorded in run metadata.

6. Recommended Run Commands

- Run training/evaluation (example):

```powershell
python -m scripts.run_training --dataset demo_dataset --config configs/demo.yaml
```

- Compute analysis artifacts from results (example):

```powershell
python scripts\_run_validate_demo.py
```

7. Output Artifacts

- `results/<dataset>/raw_results.csv`
- `results/<dataset>/normalized_results.csv`
- `results/<dataset>/cbs_scores.csv`
- `results/<dataset>/cbs_validation/*`
- `results/<dataset>/significance/*`

8. Metadata

- Each experiment should write a `run_metadata.json` containing: `git_commit`, `python_version`, `requirements_hash`, `seed_list` and `analysis_parameters` to allow exact recreation of analysis runs.