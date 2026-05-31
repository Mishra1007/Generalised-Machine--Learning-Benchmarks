Benchmarking Methodology

(Author-ready section suitable for inclusion in conference or journal manuscripts.)

Introduction

We present a reproducible benchmarking framework designed to evaluate machine learning models across multiple datasets using principled statistical methodology and a composite scoring mechanism (Composite Benchmark Score, CBS). The framework emphasizes paired comparisons, effect sizes, and stability analyses to support scientifically defensible conclusions.

Experimental Design

- Validation strategy: Repeated k-fold cross-validation with explicit `repetition_id` and `fold_id` identifiers to enable paired statistical tests that account for within-dataset variance.
- Metric collection: Primary and auxiliary per-fold metrics are recorded. Where necessary, metrics are inverted (e.g., latency) to align higher-is-better semantics prior to normalization.
- Normalization: Per-dataset min-max normalization (configurable) is applied to component metrics so that the CBS is interpretable as an aggregate on a common scale.

Composite Benchmark Score (CBS)

- Definition: CBS is a weighted linear combination of normalized component metrics, using a canonical weight vector defined in `metrics/cbs.py`.
- Rationale: Aggregation supports ranking and simplifies multi-criteria comparison while retaining the ability to decompose the aggregate into component contributions.
- Validation: We validate the CBS by: (1) leave-one-out influence, (2) sensitivity to perturbations in each metric, (3) weight robustness, and (4) Monte Carlo stability tests to quantify the reliability of top-ranked models.

Statistical Analysis

- Global differences across models are tested using the Friedman test. Where the Friedman test rejects the null hypothesis, we apply the Nemenyi post-hoc to identify groups of models that are not significantly different.
- Pairwise comparisons are performed using the Wilcoxon Signed-Rank test with Holm-Bonferroni correction for multiple testing. Each pairwise result is augmented with Cohen's d (paired) and Cliff's Delta to convey effect size and practical significance.
- Confidence intervals for mean differences are provided via both parametric t-based intervals and bootstrap percentile intervals (configurable number of bootstrap iterations).

Reporting and Interpretation

- Results are published as CSV artifacts containing full comparison matrices and a narrative `significance_report.md` that highlights statistically and practically significant differences.
- Authors are advised to prioritize effect sizes and stability analyses when drawing conclusions rather than relying solely on p-values.

Reproducibility

- The analysis layer operates on stored artifacts so that statistical reporting can be reproduced without retraining models. All analysis parameters (seeds, bootstrap iterations, MC iterations) are explicit and recorded.

Conclusion

This methodology combines rigorous statistical testing with sensitivity and stability analyses for composite scores. The approach is designed to be conservative in claims but informative for practitioners and researchers who need robust comparative evidence.