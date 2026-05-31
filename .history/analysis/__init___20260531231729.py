"""Analysis module for statistical testing, effect sizes, and reporting."""

from analysis.effect_size import (
    cliffs_delta,
    cohens_d,
    effect_size_summary,
    interpret_cliffs_delta,
    interpret_cohens_d,
)
from analysis.reports import build_significance_report, write_significance_artifacts
from analysis.significance import (
    global_significance_analysis,
    nemenyi_critical_difference,
    nemenyi_test,
    pairwise_wilcoxon,
)
from analysis.statistics import (
    bootstrap_confidence_interval,
    friedman_test,
    mean_confidence_interval,
    rank_models,
    wilcoxon_signed_rank,
)

__all__ = [
    'bootstrap_confidence_interval',
    'build_significance_report',
    'cliffs_delta',
    'cohens_d',
    'effect_size_summary',
    'friedman_test',
    'global_significance_analysis',
    'interpret_cliffs_delta',
    'interpret_cohens_d',
    'mean_confidence_interval',
    'nemenyi_critical_difference',
    'nemenyi_test',
    'pairwise_wilcoxon',
    'rank_models',
    'wilcoxon_signed_rank',
    'write_significance_artifacts',
]
"""Analysis module for statistical testing and reporting."""
