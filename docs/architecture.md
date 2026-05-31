System Architecture

Overview

This document describes the logical and physical architecture of the Generalised Machine Learning Benchmarks framework used to evaluate and compare model implementations across datasets and experimental settings. The architecture is modular, file-based, and designed for reproducible benchmarking and statistical analysis.

Components

- Metrics Engine: Implements per-fold and aggregated metrics. Located in `metrics/`. Key modules:
  - `metrics/pipeline.py`: orchestrates metric calculation per fold and aggregation policies.
  - `metrics/cbs.py`: implements the Composite Benchmark Score (CBS) calculation and weight constants; authoriative source for score composition.
  - `metrics/normalization.py`: normalization utilities applied prior to CBS computation.

- Models Engine: Model definitions, training and inference wrappers under `models/`.

- Data Loaders & Preprocessing: Dataset loaders and preprocessing pipelines in `datasets/` and `preprocessing/`.

- Validation & Fold Management: Cross-validation and fold orchestration in `validation/` (e.g., `cross_validator.py`, `fold_manager.py`).

- Storage & Artifacts: File-based result storage under `results/<dataset>/` with consistent artifact names: `raw_results.csv`, `normalized_results.csv`, `cbs_scores.csv`, and `plots/`.

- Analysis Layer: Additive analysis modules in `analysis/` that consume stored artifacts to run inferential statistics and validation analysis without modifying metric calculation code. Key modules introduced:
  - `analysis/statistics.py`: Friedman, Wilcoxon, ranking and confidence interval helpers.
  - `analysis/effect_size.py`: Cohen's d, Cliff's Delta.
  - `analysis/significance.py`: Nemenyi post-hoc, pairwise Wilcoxon orchestration.
  - `analysis/reports.py`: report and CSV writers for significance artifacts.
  - `analysis/cbs_validation.py`: CBS robustness and sensitivity analyses and plotting.

- CLI/Utilities: Lightweight scripts under `scripts/` to run demos, analysis, and the CBS validator.

Orchestration Flow

1. Dataset preparation and preprocessing produce fold-level input artifacts consumed by the Metrics Engine.
2. Metrics Engine runs per-fold evaluation and writes `raw_results.csv` containing per-fold scores and metadata identifiers (e.g., `repetition_id`, `fold_id`, `model`).
3. Normalization is applied (via `metrics/normalization.py`) producing `normalized_results.csv`.
4. CBS scores are computed (via `metrics/cbs.py`) and persisted as `cbs_scores.csv`.
5. Analysis layer consumes these persisted files to perform statistical tests and validation (no re-execution of model training required).
6. Analysis writes artifacts and human-readable reports under `results/<dataset>/` (e.g., `results/demo_dataset_storage/cbs_validation/`).

Data Flow

- Input: Raw datasets and model artifacts.
- Intermediate: Per-fold metric outputs (`raw_results.csv`) with aligned keys (`repetition_id`, `fold_id`) to support paired tests.
- Normalized: `normalized_results.csv` (min-max/other normalization per dataset as configured).
- CBS: `cbs_scores.csv` containing model-level CBS and component contributions.
- Analysis outputs: CSVs and `*.md` reports plus plots under dataset `cbs_validation/` and global `significance/` artifact directories.

Storage Flow

- Canonical artifact locations:
  - `results/<dataset>/raw_results.csv`
  - `results/<dataset>/normalized_results.csv`
  - `results/<dataset>/cbs_scores.csv`
  - `results/<dataset>/cbs_validation/` (CSV reports + `plots/`)
  - `results/<dataset>/significance/` (statistical comparison artifacts)

- Artifacts are human-readable CSV and Markdown files; plots are saved in PNG and PDF.

Evaluation Pipeline

- Fold-level metrics calculated and stored.
- Aggregation into model-level summaries and computation of CBS.
- Statistical comparison pipeline (analysis layer):
  - Build aligned score matrix across models using (`repetition_id`, `fold_id`) keys to ensure valid paired tests.
  - Run Friedman test for global differences.
  - If global effect detected, run Nemenyi post-hoc for multi-way pairwise comparisons.
  - Run pairwise Wilcoxon Signed-Rank tests with effect-size augmentation (Cohen's d, Cliff's Delta) and bootstrap confidence intervals for means.
  - Produce ranking tables, comparison matrices, and narrative `significance_report.md` artifacts.

Notes

- The analysis layer is intentionally additive and does not modify the metric computation source-of-truth files (e.g., `metrics/cbs.py`).
- All orchestration is file-driven to support reproducible re-runs and independent analysis processes.