Research Methodology

Purpose

This document captures the research-level methodology used to design, run, and analyse the benchmarks. It provides the conceptual rationale and operational details necessary for peer reviewers and researchers to understand experimental choices.

Benchmarking Philosophy

- Comparative, reproducible, and statistically rigorous: The framework evaluates models on common datasets under controlled preprocessing and validation splits, emphasizing paired comparisons to reduce variance from data splits.
- Transparent scoring: Metrics are stored as raw fold-level outputs; normalization and composition into the Composite Benchmark Score (CBS) are explicit and auditable.
- Separation of concerns: Metric computation, aggregation, and statistical analysis are distinct stages to make provenance and reuse explicit.

Model Evaluation Process

- Data splits and validation: Models are evaluated using repeated cross-validation with explicit identifiers (`repetition_id`, `fold_id`) to permit paired tests. Fold management is implemented in `validation/`.
- Metric collection: Per-fold metrics (primary metric and auxiliary metrics) are written to `raw_results.csv`.
- Normalization: Per-dataset normalization is applied to place component metrics on a comparable scale (see `metrics/normalization.py`).
- CBS computation: `metrics/cbs.py` defines CBS components and weights. CBS is computed after normalization and saved to `cbs_scores.csv`.

Statistical Validation Process

- Pairing: All paired tests align fold-level observations by `repetition_id` and `fold_id`. Unpaired or mismatched folds are dropped prior to tests.
- Global test: Use the Friedman test for repeated-measures non-parametric ANOVA across models.
- Post-hoc: If Friedman is significant, use the Nemenyi test to report critical differences and identify sets of indistinguishable models.
- Pairwise confirmation: Wilcoxon Signed-Rank tests for pairwise comparisons with Holm-Bonferroni correction for multiple comparisons.
- Confidence intervals: For mean differences we report both parametric mean +/- t CI and bootstrap percentile CIs.
- Effect sizes: Report Cohen's d (paired variant) and Cliff's Delta with interpretation thresholds to quantify practical significance beyond p-values.

CBS Methodology

- Description: CBS is a weighted linear composite of normalized component metrics (see `metrics/cbs.py` for the canonical weight vector). Each component represents a distinct quality axis (e.g., accuracy, stability, efficiency).
- Rationale: Compositional scoring enables a single scalar to support ranking and multi-criteria analysis while preserving component-level interpretability via contribution breakdowns.
- Validation: The analysis layer contains a dedicated CBS validation module (`analysis/cbs_validation.py`) that performs:
  - Sensitivity analysis: perturbation of single components to quantify CBS responsiveness.
  - Weight robustness: perturbation of CBS weights to measure ranking stability.
  - Metric dominance and leave-one-out influence analyses.
  - Correlation and redundancy checks between components and CBS.
  - Monte Carlo stability: noisy perturbations simulating measurement error to estimate ranking retention probabilities.

Reporting

- Artifact set: `significance_report.md`, `comparison_table.csv`, `ranking_table.csv`, and the `cbs_validation` bundle (CSV + plots).
- Interpretation guidance: All statistical outputs include interpretation notes and thresholds (e.g., small/medium/large effect-size bands) to guide authors in writing results sections.

Ethical and methodological transparency

- All decisions about normalization, component selection, and CBS weight choices are documented and preserved in repository code and `docs/` to ensure reviewers can audit and, if desired, reproduce alternative choices.