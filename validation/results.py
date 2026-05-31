"""
Validation results storage and access.

Stores fold-level and aggregated results with full provenance.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    """
    Result from a single fold evaluation.
    
    Stores all information about a single fold's training and testing.
    """
    repetition_id: int
    fold_id: int
    model_name: str
    dataset_name: str
    metrics: Dict[str, float]
    train_size: int
    test_size: int
    train_time: float = 0.0
    eval_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Optional: store predictions for analysis
    y_test: Optional[np.ndarray] = None
    y_pred: Optional[np.ndarray] = None
    y_pred_proba: Optional[np.ndarray] = None
    test_indices: Optional[List[int]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        # Convert numpy arrays to lists for serialization
        if self.y_test is not None:
            data['y_test'] = self.y_test.tolist()
        if self.y_pred is not None:
            data['y_pred'] = self.y_pred.tolist()
        if self.y_pred_proba is not None:
            data['y_pred_proba'] = self.y_pred_proba.tolist()
        return data


class ValidationResults:
    """
    Accumulate and manage validation results.
    
    Stores results for all folds and provides aggregation.
    """
    
    def __init__(self, model_name: str, dataset_name: str, random_state: int = 42):
        """
        Initialize validation results storage.
        
        Args:
            model_name: Name of the model being evaluated
            dataset_name: Name of the dataset
            random_state: Random seed used for validation
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.random_state = random_state
        self.fold_results: List[FoldResult] = []
        self.start_time = datetime.now()
        
        logger.info(
            f"ValidationResults created for model={model_name}, "
            f"dataset={dataset_name}, seed={random_state}"
        )
    
    def add_fold_result(self, fold_result: FoldResult) -> None:
        """
        Add result from a single fold.
        
        Args:
            fold_result: FoldResult object with all fold information
        """
        self.fold_results.append(fold_result)
        logger.debug(
            f"Added fold result: Rep {fold_result.repetition_id}, "
            f"Fold {fold_result.fold_id}, "
            f"Accuracy: {fold_result.metrics.get('accuracy', 0):.4f}"
        )
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert results to pandas DataFrame.
        
        Returns:
            DataFrame with one row per fold
        """
        data = []
        for result in self.fold_results:
            row = {
                'repetition_id': result.repetition_id,
                'fold_id': result.fold_id,
                'model_name': result.model_name,
                'dataset_name': result.dataset_name,
                'train_size': result.train_size,
                'test_size': result.test_size,
                'train_time': result.train_time,
                'eval_time': result.eval_time,
                'timestamp': result.timestamp,
            }
            # Add metrics
            row.update(result.metrics)
            data.append(row)
        
        return pd.DataFrame(data)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics across all folds.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.fold_results:
            raise RuntimeError("No fold results to summarize")
        
        df = self.to_dataframe()
        
        # Get metric columns (exclude metadata)
        metric_cols = [
            col for col in df.columns
            if col not in [
                'repetition_id', 'fold_id', 'model_name', 'dataset_name',
                'train_size', 'test_size', 'train_time', 'eval_time', 'timestamp'
            ]
        ]
        
        summary = {
            'model_name': self.model_name,
            'dataset_name': self.dataset_name,
            'total_folds': len(self.fold_results),
            'n_repetitions': len(set(r.repetition_id for r in self.fold_results)),
            'folds_per_repetition': len(set(r.fold_id for r in self.fold_results)),
            'total_train_samples': sum(r.train_size for r in self.fold_results),
            'total_eval_samples': sum(r.test_size for r in self.fold_results),
            'total_train_time': sum(r.train_time for r in self.fold_results),
            'total_eval_time': sum(r.eval_time for r in self.fold_results),
        }
        
        # Add metric statistics
        for metric in metric_cols:
            values = df[metric].dropna()
            if len(values) > 0:
                summary[f'{metric}_mean'] = float(values.mean())
                summary[f'{metric}_std'] = float(values.std())
                summary[f'{metric}_min'] = float(values.min())
                summary[f'{metric}_max'] = float(values.max())
        
        return summary
    
    def get_repetition_summary(self) -> pd.DataFrame:
        """
        Get summary per repetition (average across folds per repetition).
        
        Returns:
            DataFrame with one row per repetition
        """
        df = self.to_dataframe()
        
        # Get metric columns
        metric_cols = [
            col for col in df.columns
            if col not in [
                'repetition_id', 'fold_id', 'model_name', 'dataset_name',
                'train_size', 'test_size', 'train_time', 'eval_time', 'timestamp'
            ]
        ]
        
        # Group by repetition
        summaries = []
        for rep_id in sorted(df['repetition_id'].unique()):
            rep_data = df[df['repetition_id'] == rep_id]
            rep_dict = {
                'repetition_id': rep_id,
                'model_name': self.model_name,
                'dataset_name': self.dataset_name,
                'n_folds': len(rep_data),
            }
            
            for metric in metric_cols:
                values = rep_data[metric].dropna()
                if len(values) > 0:
                    rep_dict[f'{metric}_mean'] = float(values.mean())
                    rep_dict[f'{metric}_std'] = float(values.std())
            
            summaries.append(rep_dict)
        
        return pd.DataFrame(summaries)
    
    def get_fold_summary(self) -> pd.DataFrame:
        """
        Get summary per fold position (average across repetitions per fold).
        
        Returns:
            DataFrame with one row per fold position
        """
        df = self.to_dataframe()
        
        # Get metric columns
        metric_cols = [
            col for col in df.columns
            if col not in [
                'repetition_id', 'fold_id', 'model_name', 'dataset_name',
                'train_size', 'test_size', 'train_time', 'eval_time', 'timestamp'
            ]
        ]
        
        # Group by fold
        summaries = []
        for fold_id in sorted(df['fold_id'].unique()):
            fold_data = df[df['fold_id'] == fold_id]
            fold_dict = {
                'fold_id': fold_id,
                'model_name': self.model_name,
                'dataset_name': self.dataset_name,
                'n_repetitions': len(fold_data),
            }
            
            for metric in metric_cols:
                values = fold_data[metric].dropna()
                if len(values) > 0:
                    fold_dict[f'{metric}_mean'] = float(values.mean())
                    fold_dict[f'{metric}_std'] = float(values.std())
            
            summaries.append(fold_dict)
        
        return pd.DataFrame(summaries)
    
    def get_best_fold(self, metric: str = 'accuracy') -> FoldResult:
        """
        Get fold with best performance on given metric.
        
        Args:
            metric: Metric name to use for ranking
        
        Returns:
            FoldResult with highest metric value
        """
        best_result = max(
            self.fold_results,
            key=lambda r: r.metrics.get(metric, float('-inf'))
        )
        return best_result
    
    def get_worst_fold(self, metric: str = 'accuracy') -> FoldResult:
        """
        Get fold with worst performance on given metric.
        
        Args:
            metric: Metric name to use for ranking
        
        Returns:
            FoldResult with lowest metric value
        """
        worst_result = min(
            self.fold_results,
            key=lambda r: r.metrics.get(metric, float('inf'))
        )
        return worst_result
    
    def log_summary(self) -> None:
        """Log summary of validation results."""
        summary = self.get_summary()
        
        logger.info("\n" + "=" * 60)
        logger.info(f"VALIDATION SUMMARY: {self.model_name} on {self.dataset_name}")
        logger.info("=" * 60)
        logger.info(f"Total folds: {summary['total_folds']}")
        logger.info(f"Repetitions: {summary['n_repetitions']}")
        logger.info(f"Folds per rep: {summary['folds_per_repetition']}")
        
        # Log metrics
        for key, value in summary.items():
            if isinstance(value, float):
                logger.info(f"{key}: {value:.4f}")
        
        logger.info("=" * 60 + "\n")
