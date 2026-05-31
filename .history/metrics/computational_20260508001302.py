"""Computational and efficiency metrics."""

import logging
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


def compute_timing_metrics(
    train_time: float,
    eval_time: float,
    train_samples: int = None,
    eval_samples: int = None,
) -> dict:
    """
    Compute timing and computational efficiency metrics.
    
    Args:
        train_time: Training time in seconds
        eval_time: Evaluation/inference time in seconds
        train_samples: Number of training samples (optional, for per-sample metrics)
        eval_samples: Number of evaluation samples (optional, for per-sample metrics)
    
    Returns:
        Dictionary with timing metrics:
        - train_time: Raw training time
        - eval_time: Raw inference time
        - total_time: Sum of train and eval time
        - train_time_per_sample: Training time per sample (if samples provided)
        - eval_time_per_sample: Evaluation time per sample (if samples provided)
        - time_ratio: Inference/Training time ratio
    
    Example:
        >>> metrics = compute_timing_metrics(train_time=0.5, eval_time=0.1, train_samples=100, eval_samples=30)
        >>> print(metrics['train_time_per_sample'])
    """
    metrics = {
        'train_time': train_time,
        'eval_time': eval_time,
        'total_time': train_time + eval_time,
    }
    
    # Per-sample metrics
    if train_samples is not None and train_samples > 0:
        metrics['train_time_per_sample'] = train_time / train_samples
    
    if eval_samples is not None and eval_samples > 0:
        metrics['eval_time_per_sample'] = eval_time / eval_samples
    
    # Time ratio (for comparison)
    if train_time > 0:
        metrics['time_ratio'] = eval_time / train_time
    else:
        metrics['time_ratio'] = 0.0
    
    return metrics


def aggregate_timing_metrics(timing_metrics_list: List[dict]) -> dict:
    """
    Aggregate timing metrics across multiple folds.
    
    Args:
        timing_metrics_list: List of timing metrics dictionaries from multiple folds
    
    Returns:
        Dictionary with aggregated timing metrics:
        - mean_train_time, std_train_time, min/max_train_time
        - mean_eval_time, std_eval_time, min/max_eval_time
        - mean_total_time
    
    Example:
        >>> timings = [
        ...     {'train_time': 0.5, 'eval_time': 0.1},
        ...     {'train_time': 0.6, 'eval_time': 0.09},
        ... ]
        >>> agg = aggregate_timing_metrics(timings)
    """
    if not timing_metrics_list:
        return {}
    
    train_times = [m['train_time'] for m in timing_metrics_list]
    eval_times = [m['eval_time'] for m in timing_metrics_list]
    total_times = [m['total_time'] for m in timing_metrics_list]
    
    aggregated = {
        # Training time
        'train_time_mean': np.mean(train_times),
        'train_time_std': np.std(train_times),
        'train_time_min': np.min(train_times),
        'train_time_max': np.max(train_times),
        'train_time_total': np.sum(train_times),
        
        # Evaluation time
        'eval_time_mean': np.mean(eval_times),
        'eval_time_std': np.std(eval_times),
        'eval_time_min': np.min(eval_times),
        'eval_time_max': np.max(eval_times),
        'eval_time_total': np.sum(eval_times),
        
        # Total time
        'total_time_mean': np.mean(total_times),
        'total_time_std': np.std(total_times),
        'total_time_min': np.min(total_times),
        'total_time_max': np.max(total_times),
        'total_time_total': np.sum(total_times),
        
        # Count
        'n_folds': len(timing_metrics_list),
    }
    
    return aggregated


def compute_computational_efficiency_metrics(
    aggregated_timing: dict,
    n_repetitions: int = 1,
) -> dict:
    """
    Compute efficiency metrics from aggregated timing.
    
    Args:
        aggregated_timing: Dictionary from aggregate_timing_metrics()
        n_repetitions: Number of CV repetitions (for context)
    
    Returns:
        Dictionary with efficiency metrics:
        - avg_train_time_per_fold
        - avg_eval_time_per_fold
        - efficiency_score (inverse of average time)
    
    Example:
        >>> efficiency = compute_computational_efficiency_metrics(agg_timing, n_repetitions=3)
    """
    if not aggregated_timing or 'train_time_mean' not in aggregated_timing:
        return {}
    
    metrics = {
        'avg_train_time_per_fold': aggregated_timing.get('train_time_mean', 0),
        'avg_eval_time_per_fold': aggregated_timing.get('eval_time_mean', 0),
        'avg_total_time_per_fold': aggregated_timing.get('total_time_mean', 0),
    }
    
    # Efficiency score: inverse of average time (higher is better)
    total_mean = metrics['avg_total_time_per_fold']
    if total_mean > 0:
        # Efficiency score between 0 and 1: 1/(1 + avg_time)
        metrics['efficiency_score'] = 1.0 / (1.0 + total_mean)
    else:
        metrics['efficiency_score'] = 1.0
    
    # Total computational cost
    n_folds = aggregated_timing.get('n_folds', 1)
    metrics['total_computational_cost'] = aggregated_timing.get('total_time_total', 0)
    
    return metrics


def compare_computational_efficiency(
    timing_results: Dict[str, dict],
) -> dict:
    """
    Compare computational efficiency across multiple models.
    
    Args:
        timing_results: Dict mapping model_name to aggregated_timing dict
    
    Returns:
        Dictionary with ranked model efficiency
    
    Example:
        >>> timing_results = {
        ...     'rf': {'train_time_mean': 0.5, 'eval_time_mean': 0.1},
        ...     'svm': {'train_time_mean': 2.0, 'eval_time_mean': 0.5},
        ... }
        >>> comparison = compare_computational_efficiency(timing_results)
    """
    if not timing_results:
        return {}
    
    comparison = {}
    for model_name, timing in timing_results.items():
        efficiency = compute_computational_efficiency_metrics(timing)
        comparison[model_name] = {
            'avg_train_time': timing.get('train_time_mean', 0),
            'avg_eval_time': timing.get('eval_time_mean', 0),
            'avg_total_time': timing.get('total_time_mean', 0),
            'efficiency_score': efficiency.get('efficiency_score', 0),
        }
    
    return comparison
