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

__all__ = [
    'compute_classification_metrics',
    'compute_fold_metrics',
    'get_confusion_matrix',
    'compute_per_class_metrics',
    'MetricAggregator',
    'aggregate_predictions',
    'compute_confidence_intervals',
]
