"""Metrics module for evaluation and scoring."""

from metrics.scorers import (
    compute_classification_metrics,
    compute_fold_metrics,
    get_confusion_matrix,
    compute_per_class_metrics,
)
from metrics.aggregators import (
    MetricAggregator,
    aggregate_predictions,
    compute_confidence_intervals,
)
from metrics.computational import (
    compute_timing_metrics,
    aggregate_timing_metrics,
    compute_computational_efficiency_metrics,
    compare_computational_efficiency,
)
from metrics.stability import (
    compute_stability_metrics,
    compute_all_stability_metrics,
    assess_model_stability,
    extract_metric_across_folds,
)
from metrics.pipeline import (
    MetricsCalculator,
    MetricsValidator,
)

__all__ = [
    # Predictive metrics
    'compute_classification_metrics',
    'compute_fold_metrics',
    'get_confusion_matrix',
    'compute_per_class_metrics',
    
    # Aggregation
    'MetricAggregator',
    'aggregate_predictions',
    'compute_confidence_intervals',
    
    # Computational metrics
    'compute_timing_metrics',
    'aggregate_timing_metrics',
    'compute_computational_efficiency_metrics',
    'compare_computational_efficiency',
    
    # Stability metrics
    'compute_stability_metrics',
    'compute_all_stability_metrics',
    'assess_model_stability',
    'extract_metric_across_folds',
    
    # Unified pipeline
    'MetricsCalculator',
    'MetricsValidator',
]
