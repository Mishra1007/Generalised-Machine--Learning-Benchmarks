Threats to Validity

This document enumerates threats to validity and mitigation strategies across internal, external, construct, and conclusion validity.

1. Internal Validity

- Implementation Bias: Bugs in metric calculations or CBS composition can bias results. Mitigation: keep `metrics/cbs.py` and normalization code as authoritative, add unit tests (existing `tests/`), and perform independent re-computation of CBS in the analysis layer for cross-checks.

- Randomness Effects: Non-deterministic training and data splits introduce variability. Mitigation: use seeded RNGs for splits and training when possible; report run-to-run variability and use repeated CV.

- Data Leakage: Improper preprocessing (e.g., global normalization across train+test) can inflate results. Mitigation: apply preprocessing only using training-fold statistics and save transformers per-fold.

2. External Validity

- Dataset Representativeness: Datasets in the benchmark may not cover the operational regime of real-world tasks. Mitigation: document dataset characteristics and avoid over-generalizing conclusions.

- Model Generalization: Evaluated models may not be representative of the broader model family. Mitigation: explicitly state selection criteria and include a diverse set of representative baselines.

3. Construct Validity

- Metric Selection and Weighting: CBS depends on chosen component metrics and weight vector; poor choices can misrepresent priorities. Mitigation: justify metric choices academically, provide ablation (leave-one-out) and sensitivity analyses, and make weights configurable.

- CBS Assumptions: CBS assumes meaningful normalization and linear combination of metrics. Mitigation: provide extensive normalization validation, ranking distortion checks, and weight robustness analyses (provided in `analysis/cbs_validation.py`).

4. Conclusion Validity

- Statistical Power: Small sample sizes or few folds reduce power. Mitigation: use repeated CV, report confidence intervals and effect sizes, and be conservative in claims when power is low.

- Multiple Comparisons: Many pairwise tests increase false-positive risk. Mitigation: use global tests (Friedman) and correct for multiple comparisons (Holm-Bonferroni or equivalent) and emphasize effect sizes over isolated p-values.

Summary

- The repository provides mitigation scaffolding (paired tests, effect sizes, bootstrap CIs, Monte Carlo stability) to reduce common threats. All mitigation steps should be reported transparently in manuscripts and supplementary materials.