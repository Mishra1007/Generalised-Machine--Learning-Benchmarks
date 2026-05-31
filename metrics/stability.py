"""Stability metrics for model evaluation."""

import logging
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


def compute_stability_metrics(
    metric_values: List[float],
    metric_name: str = 'metric',
) -> dict:
    """
    Compute stability metrics for a single metric across folds.
    
    Args:
        metric_values: List of metric values (one per fold)
        metric_name: Name of metric (for logging)
    
    Returns:
        Dictionary with stability metrics:
        - mean: Mean across folds
        - std: Standard deviation
        - min: Minimum value
        - max: Maximum value
        - range: Max - Min
        - coefficient_of_variation: Std / Mean
        - stability_score: 1 / (1 + coefficient_of_variation)
    
    Example:
        >>> f1_scores = [0.85, 0.87, 0.84, 0.88, 0.86]
        >>> stability = compute_stability_metrics(f1_scores, 'F1')
        >>> print(f"Mean: {stability['mean']:.4f}, Std: {stability['std']:.4f}")
    """
    if not metric_values:
        logger.warning(f"Empty metric values for {metric_name}")
        return {}
    
    values = np.array(metric_values)
    
    metrics = {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'range': float(np.max(values) - np.min(values)),
        'n_samples': len(values),
    }
    
    # Coefficient of variation (std / mean)
    if metrics['mean'] > 0:
        metrics['coefficient_of_variation'] = metrics['std'] / metrics['mean']
    else:
        metrics['coefficient_of_variation'] = 0.0
    
    # Stability score: higher is more stable (1 / (1 + cv))
    metrics['stability_score'] = 1.0 / (1.0 + metrics['coefficient_of_variation'])
    
    return metrics


def extract_metric_across_folds(
    fold_metrics_list: List[dict],
    metric_name: str,
) -> List[float]:
    """
    Extract a single metric from fold-level results.
    
    Args:
        fold_metrics_list: List of metrics dictionaries from folds
        metric_name: Name of metric to extract
    
    Returns:
        List of metric values
    
    Example:
        >>> folds = [
        ...     {'accuracy': 0.90, 'f1': 0.85},
        ...     {'accuracy': 0.92, 'f1': 0.87},
        ... ]
        >>> f1_values = extract_metric_across_folds(folds, 'f1')
    """
    values = []
    for fold_metrics in fold_metrics_list:
        if metric_name in fold_metrics:
            values.append(fold_metrics[metric_name])
        else:
            logger.warning(f"Metric '{metric_name}' not found in fold results")
    
    return values


def compute_all_stability_metrics(
    fold_metrics_list: List[dict],
    metric_names: List[str] = None,
) -> dict:
    """
    Compute stability metrics for all metrics in fold results.
    
    Args:
        fold_metrics_list: List of fold-level metrics dictionaries
        metric_names: List of metric names to analyze (None = all)
    
    Returns:
        Dictionary mapping metric_name to stability metrics
    
    Example:
        >>> folds = [
        ...     {'accuracy': 0.90, 'f1': 0.85, 'precision': 0.88},
        ...     {'accuracy': 0.92, 'f1': 0.87, 'precision': 0.90},
        ... ]
        >>> stability = compute_all_stability_metrics(folds)
    """
    if not fold_metrics_list:
        return {}
    
    # Determine which metrics to analyze
    if metric_names is None:
        # Use all keys from first fold
        metric_names = list(fold_metrics_list[0].keys())
    
    stability_metrics = {}
    
    for metric_name in metric_names:
        values = extract_metric_across_folds(fold_metrics_list, metric_name)
        
        if values:
            stability_metrics[metric_name] = compute_stability_metrics(values, metric_name)
        else:
            logger.warning(f"No values found for metric '{metric_name}'")
    
    return stability_metrics


def aggregate_stability_by_repetition(
    fold_results_by_rep: Dict[int, List[dict]],
) -> dict:
    """
    Aggregate stability metrics by repetition.
    
    Args:
        fold_results_by_rep: Dict mapping repetition_id to list of fold metrics
    
    Returns:
        Dictionary with per-repetition stability metrics
    
    Example:
        >>> by_rep = {
        ...     0: [{'accuracy': 0.90}, {'accuracy': 0.92}],
        ...     1: [{'accuracy': 0.91}, {'accuracy': 0.93}],
        ... }
        >>> stability_by_rep = aggregate_stability_by_repetition(by_rep)
    """
    if not fold_results_by_rep:
        return {}
    
    stability_by_rep = {}
    
    for rep_id, fold_metrics in fold_results_by_rep.items():
        stability_by_rep[rep_id] = compute_all_stability_metrics(fold_metrics)
    
    return stability_by_rep


def compute_cross_repetition_stability(
    metrics_by_repetition: Dict[int, dict],
    metric_names: List[str] = None,
) -> dict:
    """
    Compute stability of metrics across repetitions.
    
    Shows how consistent a metric is across different CV repetitions.
    
    Args:
        metrics_by_repetition: Dict mapping repetition_id to metrics dict
        metric_names: List of metrics to analyze
    
    Returns:
        Dictionary with cross-repetition stability
    
    Example:
        >>> by_rep = {
        ...     0: {'accuracy_mean': 0.90},
        ...     1: {'accuracy_mean': 0.91},
        ...     2: {'accuracy_mean': 0.90},
        ... }
        >>> cross_rep = compute_cross_repetition_stability(by_rep, ['accuracy_mean'])
    """
    if not metrics_by_repetition:
        return {}
    
    # Determine metrics
    if metric_names is None and metrics_by_repetition:
        # Get first rep's metrics
        first_rep = next(iter(metrics_by_repetition.values()))
        metric_names = list(first_rep.keys())
    
    cross_rep_stability = {}
    
    for metric_name in (metric_names or []):
        values = []
        for rep_metrics in metrics_by_repetition.values():
            if metric_name in rep_metrics:
                values.append(rep_metrics[metric_name])
        
        if values:
            cross_rep_stability[metric_name] = compute_stability_metrics(values, metric_name)
    
    return cross_rep_stability


def assess_model_stability(
    fold_metrics_list: List[dict],
    primary_metric: str = 'f1',
    threshold: float = 0.05,
) -> dict:
    """
    Assess overall model stability based on metric variation.
    
    Args:
        fold_metrics_list: List of fold metrics
        primary_metric: Primary metric to assess (default 'f1')
        threshold: CV threshold for stability (default 5%)
    
    Returns:
        Dictionary with stability assessment:
        - is_stable: Boolean indicating if model is stable
        - primary_metric_std: Std of primary metric
        - primary_metric_cv: Coefficient of variation
        - stability_level: 'High', 'Medium', 'Low'
    
    Example:
        >>> folds = [{'f1': 0.85}, {'f1': 0.86}, {'f1': 0.85}]
        >>> assessment = assess_model_stability(folds, 'f1', threshold=0.05)
        >>> print(assessment['stability_level'])
    """
    if not fold_metrics_list:
        return {'is_stable': False, 'stability_level': 'Unknown'}
    
    values = extract_metric_across_folds(fold_metrics_list, primary_metric)
    
    if not values:
        return {'is_stable': False, 'stability_level': 'Unknown'}
    
    stability = compute_stability_metrics(values, primary_metric)
    
    cv = stability['coefficient_of_variation']
    
    assessment = {
        'is_stable': cv <= threshold,
        'primary_metric': primary_metric,
        'primary_metric_mean': stability['mean'],
        'primary_metric_std': stability['std'],
        'primary_metric_cv': cv,
        'threshold': threshold,
    }
    
    # Determine stability level
    if cv <= threshold:
        assessment['stability_level'] = 'High'
    elif cv <= 2 * threshold:
        assessment['stability_level'] = 'Medium'
    else:
        assessment['stability_level'] = 'Low'
    
    return assessment
