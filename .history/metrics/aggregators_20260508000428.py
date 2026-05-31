"""
Metric aggregation and result summarization.

Aggregates metrics across folds and repetitions.
"""

import logging
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MetricAggregator:
    """
    Aggregate metrics across multiple folds and repetitions.
    
    Computes mean, std, min, max for each metric.
    """
    
    def __init__(self):
        """Initialize aggregator."""
        self.fold_results = []
    
    def add_fold_result(
        self,
        metrics: Dict[str, float],
        repetition_id: int,
        fold_id: int
    ) -> None:
        """
        Add metrics from a single fold.
        
        Args:
            metrics: Dictionary of metric scores
            repetition_id: Repetition identifier (0-2)
            fold_id: Fold identifier (0-4)
        """
        metrics_copy = metrics.copy()
        metrics_copy['repetition_id'] = repetition_id
        metrics_copy['fold_id'] = fold_id
        self.fold_results.append(metrics_copy)
        
        logger.debug(
            f"Added fold result: Rep={repetition_id}, Fold={fold_id}, "
            f"Accuracy={metrics.get('accuracy', np.nan):.4f}"
        )
    
    def aggregate(self) -> pd.DataFrame:
        """
        Aggregate all fold results.
        
        Returns:
            DataFrame with aggregated metrics (mean, std, min, max)
        """
        if not self.fold_results:
            raise RuntimeError("No fold results to aggregate")
        
        df = pd.DataFrame(self.fold_results)
        
        # Get metric columns (exclude metadata)
        metric_cols = [
            col for col in df.columns
            if col not in ['repetition_id', 'fold_id', 'n_samples', 'n_classes']
        ]
        
        # Aggregate across all folds and repetitions
        summary = {}
        for metric in metric_cols:
            values = df[metric].dropna()
            if len(values) > 0:
                summary[f'{metric}_mean'] = values.mean()
                summary[f'{metric}_std'] = values.std()
                summary[f'{metric}_min'] = values.min()
                summary[f'{metric}_max'] = values.max()
        
        return pd.DataFrame([summary])
    
    def aggregate_by_repetition(self) -> pd.DataFrame:
        """
        Aggregate metrics by repetition (average across folds per repetition).
        
        Returns:
            DataFrame with one row per repetition
        """
        if not self.fold_results:
            raise RuntimeError("No fold results to aggregate")
        
        df = pd.DataFrame(self.fold_results)
        
        # Get metric columns
        metric_cols = [
            col for col in df.columns
            if col not in ['repetition_id', 'fold_id', 'n_samples', 'n_classes']
        ]
        
        # Group by repetition
        repetition_stats = []
        for rep_id in sorted(df['repetition_id'].unique()):
            rep_data = df[df['repetition_id'] == rep_id]
            rep_dict = {'repetition_id': rep_id}
            
            for metric in metric_cols:
                values = rep_data[metric].dropna()
                if len(values) > 0:
                    rep_dict[f'{metric}_mean'] = values.mean()
                    rep_dict[f'{metric}_std'] = values.std()
            
            repetition_stats.append(rep_dict)
        
        return pd.DataFrame(repetition_stats)
    
    def aggregate_by_fold(self) -> pd.DataFrame:
        """
        Aggregate metrics by fold across all repetitions.
        
        Returns:
            DataFrame with one row per fold position
        """
        if not self.fold_results:
            raise RuntimeError("No fold results to aggregate")
        
        df = pd.DataFrame(self.fold_results)
        
        # Get metric columns
        metric_cols = [
            col for col in df.columns
            if col not in ['repetition_id', 'fold_id', 'n_samples', 'n_classes']
        ]
        
        # Group by fold
        fold_stats = []
        for fold_id in sorted(df['fold_id'].unique()):
            fold_data = df[df['fold_id'] == fold_id]
            fold_dict = {'fold_id': fold_id}
            
            for metric in metric_cols:
                values = fold_data[metric].dropna()
                if len(values) > 0:
                    fold_dict[f'{metric}_mean'] = values.mean()
                    fold_dict[f'{metric}_std'] = values.std()
            
            fold_stats.append(fold_dict)
        
        return pd.DataFrame(fold_stats)
    
    def get_summary_statistics(self) -> Dict[str, Tuple[float, float]]:
        """
        Get summary statistics for all metrics.
        
        Returns:
            Dictionary mapping metric_name to (mean, std)
        """
        df = self.aggregate()
        
        summary = {}
        for col in df.columns:
            if '_mean' in col:
                metric_name = col.replace('_mean', '')
                mean_val = df[col].iloc[0]
                std_col = f'{metric_name}_std'
                std_val = df[std_col].iloc[0] if std_col in df.columns else 0.0
                summary[metric_name] = (mean_val, std_val)
        
        return summary
    
    def get_fold_results_dataframe(self) -> pd.DataFrame:
        """
        Get all fold results as DataFrame.
        
        Returns:
            DataFrame with all fold-level metrics
        """
        return pd.DataFrame(self.fold_results)
    
    def clear(self) -> None:
        """Clear all stored results."""
        self.fold_results = []


def aggregate_predictions(
    fold_results: List[Dict]
) -> Dict:
    """
    Aggregate predictions across all folds.
    
    Args:
        fold_results: List of fold result dictionaries
    
    Returns:
        Aggregated predictions and metrics
    """
    if not fold_results:
        raise ValueError("No fold results to aggregate")
    
    # Collect all predictions
    all_indices = []
    all_y_true = []
    all_y_pred = []
    
    for result in fold_results:
        if 'test_indices' in result:
            all_indices.extend(result['test_indices'])
            all_y_true.extend(result['y_true'])
            all_y_pred.extend(result['y_pred'])
    
    return {
        'test_indices': all_indices,
        'y_true': np.array(all_y_true),
        'y_pred': np.array(all_y_pred),
        'n_folds': len(fold_results),
    }


def compute_confidence_intervals(
    values: np.ndarray,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute confidence interval for a metric.
    
    Args:
        values: Array of metric values
        confidence: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    mean = np.mean(values)
    std_err = np.std(values, ddof=1) / np.sqrt(len(values))
    
    # Critical value for t-distribution (approximate)
    z = 1.96 if confidence == 0.95 else 2.576
    
    margin = z * std_err
    return mean, mean - margin, mean + margin
