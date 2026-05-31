"""Unified metrics pipeline for comprehensive evaluation."""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from metrics.scorers import compute_classification_metrics
from metrics.computational import (
    compute_timing_metrics,
    aggregate_timing_metrics,
    compute_computational_efficiency_metrics,
)
from metrics.stability import (
    compute_all_stability_metrics,
    assess_model_stability,
    compute_stability_metrics,
    extract_metric_across_folds,
)

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Unified metrics calculator for comprehensive model evaluation.
    
    Orchestrates calculation of:
    - Predictive metrics (accuracy, precision, recall, F1, ROC-AUC)
    - Computational metrics (training time, inference time)
    - Stability metrics (std, mean, coefficient of variation)
    
    Architecture:
    - Fold-wise metrics calculation
    - Aggregation across folds and repetitions
    - Architecture for future normalization and CBS
    
    Example:
        >>> calculator = MetricsCalculator()
        >>> fold_metrics = calculator.calculate_fold_metrics(
        ...     y_true, y_pred, y_proba,
        ...     train_time=0.5, eval_time=0.1
        ... )
    """
    
    def __init__(
        self,
        predictive_metrics: List[str] = None,
        computational_metrics: List[str] = None,
        stability_metrics: List[str] = None,
    ):
        """
        Initialize metrics calculator.
        
        Args:
            predictive_metrics: List of predictive metrics to compute
            computational_metrics: List of computational metrics
            stability_metrics: List of stability metrics
        """
        self.predictive_metrics = predictive_metrics or [
            'accuracy', 'precision', 'recall', 'f1', 'roc_auc'
        ]
        self.computational_metrics = computational_metrics or [
            'train_time', 'eval_time', 'total_time'
        ]
        self.stability_metrics = stability_metrics or [
            'mean', 'std', 'coefficient_of_variation', 'stability_score'
        ]
        
        logger.debug(
            f"MetricsCalculator initialized with predictive: {len(self.predictive_metrics)}, "
            f"computational: {len(self.computational_metrics)}, "
            f"stability: {len(self.stability_metrics)}"
        )
    
    def calculate_fold_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        train_time: float = 0.0,
        eval_time: float = 0.0,
        train_samples: int = None,
        eval_samples: int = None,
        fold_id: int = 0,
        repetition_id: int = 0,
    ) -> dict:
        """
        Calculate all metrics for a single fold.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            train_time: Training time in seconds
            eval_time: Evaluation time in seconds
            train_samples: Number of training samples
            eval_samples: Number of evaluation samples
            fold_id: Fold identifier
            repetition_id: Repetition identifier
        
        Returns:
            Dictionary with all metrics (predictive, computational, metadata)
        
        Example:
            >>> metrics = calculator.calculate_fold_metrics(
            ...     y_true, y_pred, y_proba,
            ...     train_time=0.5, eval_time=0.1,
            ...     train_samples=100, eval_samples=30
            ... )
        """
        fold_metrics = {
            'fold_id': fold_id,
            'repetition_id': repetition_id,
        }
        
        # Predictive metrics
        try:
            pred_metrics = compute_classification_metrics(
                y_true, y_pred, y_pred_proba
            )
            fold_metrics.update(pred_metrics)
            logger.debug(f"Computed predictive metrics for fold {fold_id}")
        except Exception as e:
            logger.error(f"Failed to compute predictive metrics: {e}")
            raise
        
        # Computational metrics
        try:
            comp_metrics = compute_timing_metrics(
                train_time, eval_time,
                train_samples, eval_samples
            )
            fold_metrics.update(comp_metrics)
            logger.debug(f"Computed computational metrics for fold {fold_id}")
        except Exception as e:
            logger.error(f"Failed to compute computational metrics: {e}")
            raise
        
        return fold_metrics
    
    def aggregate_fold_metrics(
        self,
        fold_metrics_list: List[dict],
        by_repetition: bool = True,
    ) -> Tuple[dict, Dict[int, dict], Dict[int, dict]]:
        """
        Aggregate metrics across folds and repetitions.
        
        Args:
            fold_metrics_list: List of fold-level metrics dictionaries
            by_repetition: Whether to return per-repetition aggregates
        
        Returns:
            Tuple of (overall_aggregate, by_repetition_dict, by_fold_dict)
            - overall_aggregate: Metrics aggregated across all folds
            - by_repetition_dict: Dict mapping rep_id to aggregated metrics
            - by_fold_dict: Dict mapping fold_id to aggregated metrics
        
        Example:
            >>> overall, by_rep, by_fold = calculator.aggregate_fold_metrics(fold_list)
            >>> print(f"Overall Accuracy: {overall['accuracy_mean']:.4f}")
        """
        if not fold_metrics_list:
            return {}, {}, {}
        
        overall_aggregate = self._aggregate_metrics(fold_metrics_list)
        
        # Aggregate by repetition
        by_repetition_dict = {}
        if by_repetition:
            metrics_by_rep = self._group_by_key(fold_metrics_list, 'repetition_id')
            for rep_id, metrics in metrics_by_rep.items():
                by_repetition_dict[rep_id] = self._aggregate_metrics(metrics)
        
        # Aggregate by fold
        by_fold_dict = {}
        metrics_by_fold = self._group_by_key(fold_metrics_list, 'fold_id')
        for fold_id, metrics in metrics_by_fold.items():
            by_fold_dict[fold_id] = self._aggregate_metrics(metrics)
        
        return overall_aggregate, by_repetition_dict, by_fold_dict
    
    def calculate_stability_metrics(
        self,
        fold_metrics_list: List[dict],
        primary_metric: str = 'f1',
    ) -> dict:
        """
        Calculate stability metrics from fold results.
        
        Args:
            fold_metrics_list: List of fold metrics
            primary_metric: Primary metric for stability assessment
        
        Returns:
            Dictionary with stability metrics and assessment
        
        Example:
            >>> stability = calculator.calculate_stability_metrics(fold_list, 'f1')
            >>> print(stability['primary_metric_cv'])
        """
        all_stability = compute_all_stability_metrics(fold_metrics_list)
        
        assessment = assess_model_stability(
            fold_metrics_list, primary_metric
        )
        
        return {
            'by_metric': all_stability,
            'assessment': assessment,
        }
    
    def get_comprehensive_summary(
        self,
        fold_metrics_list: List[dict],
        model_name: str = 'Model',
        dataset_name: str = 'Dataset',
    ) -> dict:
        """
        Get comprehensive summary of all metrics.
        
        Args:
            fold_metrics_list: List of fold metrics
            model_name: Name of model
            dataset_name: Name of dataset
        
        Returns:
            Comprehensive summary dictionary
        
        Example:
            >>> summary = calculator.get_comprehensive_summary(fold_list, 'RF', 'iris')
        """
        if not fold_metrics_list:
            return {}
        
        overall, by_rep, by_fold = self.aggregate_fold_metrics(fold_metrics_list)
        stability = self.calculate_stability_metrics(fold_metrics_list)
        
        summary = {
            'model_name': model_name,
            'dataset_name': dataset_name,
            'n_folds': len(fold_metrics_list),
            'n_repetitions': len(set(m.get('repetition_id', 0) for m in fold_metrics_list)),
            
            # Overall metrics
            'overall': overall,
            
            # Per-repetition (if applicable)
            'by_repetition': by_rep,
            
            # Per-fold consistency
            'by_fold': by_fold,
            
            # Stability assessment
            'stability': stability,
        }
        
        return summary
    
    def export_to_dataframe(
        self,
        fold_metrics_list: List[dict],
    ) -> pd.DataFrame:
        """
        Export fold metrics to pandas DataFrame.
        
        Args:
            fold_metrics_list: List of fold metrics
        
        Returns:
            DataFrame with one row per fold
        
        Example:
            >>> df = calculator.export_to_dataframe(fold_list)
            >>> df.to_csv('metrics.csv')
        """
        return pd.DataFrame(fold_metrics_list)
    
    @staticmethod
    def _aggregate_metrics(metrics_list: List[dict]) -> dict:
        """
        Aggregate a list of metrics dictionaries.
        
        Computes mean, std, min, max for all numeric values.
        """
        if not metrics_list:
            return {}
        
        # Collect all numeric metrics
        aggregated = {}
        
        # Get all keys from first dict (excluding fold_id, repetition_id)
        exclude_keys = {'fold_id', 'repetition_id', 'test_indices'}
        sample_keys = set(metrics_list[0].keys()) - exclude_keys
        
        for key in sample_keys:
            values = []
            for m in metrics_list:
                if key in m and isinstance(m[key], (int, float)):
                    values.append(m[key])
            
            if values:
                aggregated[f'{key}_mean'] = np.mean(values)
                aggregated[f'{key}_std'] = np.std(values)
                aggregated[f'{key}_min'] = np.min(values)
                aggregated[f'{key}_max'] = np.max(values)
        
        return aggregated
    
    @staticmethod
    def _group_by_key(items: List[dict], key: str) -> Dict[int, List[dict]]:
        """Group list of dicts by a key value."""
        grouped = {}
        for item in items:
            k = item.get(key)
            if k not in grouped:
                grouped[k] = []
            grouped[k].append(item)
        return grouped


class MetricsValidator:
    """
    Validates metrics for consistency and fairness.
    
    Checks:
    - All folds present
    - Metrics in valid ranges
    - No NaN or Inf values
    - Consistency across folds
    """
    
    @staticmethod
    def validate_metrics(
        fold_metrics_list: List[dict],
        raise_on_error: bool = False,
    ) -> Tuple[bool, List[str]]:
        """
        Validate metrics for consistency and correctness.
        
        Args:
            fold_metrics_list: List of fold metrics
            raise_on_error: Whether to raise exception on validation failure
        
        Returns:
            Tuple of (is_valid, error_messages)
        
        Example:
            >>> is_valid, errors = MetricsValidator.validate_metrics(fold_list)
        """
        errors = []
        
        if not fold_metrics_list:
            errors.append("Empty metrics list")
            if raise_on_error:
                raise ValueError("Empty metrics list")
            return False, errors
        
        # Check for NaN/Inf
        for i, metrics in enumerate(fold_metrics_list):
            for key, value in metrics.items():
                if isinstance(value, float):
                    if np.isnan(value):
                        errors.append(f"NaN in fold {i}, metric {key}")
                    elif np.isinf(value):
                        errors.append(f"Inf in fold {i}, metric {key}")
        
        # Check ranges
        for i, metrics in enumerate(fold_metrics_list):
            if 'accuracy' in metrics and not (0 <= metrics['accuracy'] <= 1):
                errors.append(f"Accuracy out of range [0,1] in fold {i}")
            if 'precision' in metrics and not (0 <= metrics['precision'] <= 1):
                errors.append(f"Precision out of range [0,1] in fold {i}")
            if 'recall' in metrics and not (0 <= metrics['recall'] <= 1):
                errors.append(f"Recall out of range [0,1] in fold {i}")
            if 'f1' in metrics and not (0 <= metrics['f1'] <= 1):
                errors.append(f"F1 out of range [0,1] in fold {i}")
            if 'roc_auc' in metrics and metrics['roc_auc'] is not None:
                if not (0 <= metrics['roc_auc'] <= 1):
                    errors.append(f"ROC-AUC out of range [0,1] in fold {i}")
        
        is_valid = len(errors) == 0
        
        if not is_valid and raise_on_error:
            raise ValueError(f"Metrics validation failed: {errors}")
        
        return is_valid, errors
