"""
Classification metrics for benchmarking.

Computes standard classification evaluation metrics.
"""

import logging
from typing import Dict, Tuple, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    matthews_corrcoef,
)

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray = None,
    average: str = 'weighted'
) -> Dict[str, float]:
    """
    Compute standard classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (for ROC-AUC, shape (n_samples, n_classes))
        average: Averaging method for multi-class ('weighted', 'macro', 'micro')
    
    Returns:
        Dictionary with metric scores
    
    Metrics:
        - accuracy: Overall correctness
        - precision: True positives / (True positives + False positives)
        - recall: True positives / (True positives + False negatives)
        - f1: Harmonic mean of precision and recall
        - roc_auc: Area under ROC curve (only if y_pred_proba provided)
        - matthews_cc: Matthews correlation coefficient (correlation-like, -1 to 1)
    """
    metrics = {}
    
    # Basic metrics
    try:
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
    except Exception as e:
        logger.warning(f"Failed to compute accuracy: {e}")
        metrics['accuracy'] = np.nan
    
    try:
        metrics['precision'] = precision_score(
            y_true, y_pred, average=average, zero_division=0
        )
    except Exception as e:
        logger.warning(f"Failed to compute precision: {e}")
        metrics['precision'] = np.nan
    
    try:
        metrics['recall'] = recall_score(
            y_true, y_pred, average=average, zero_division=0
        )
    except Exception as e:
        logger.warning(f"Failed to compute recall: {e}")
        metrics['recall'] = np.nan
    
    try:
        metrics['f1'] = f1_score(
            y_true, y_pred, average=average, zero_division=0
        )
    except Exception as e:
        logger.warning(f"Failed to compute f1: {e}")
        metrics['f1'] = np.nan
    
    # ROC-AUC (only for binary or multi-class with probabilities)
    if y_pred_proba is not None:
        try:
            n_classes = len(np.unique(y_true))
            if n_classes == 2:
                # Binary classification: use second class probability
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            else:
                # Multi-class: use OvR with weighted average
                metrics['roc_auc'] = roc_auc_score(
                    y_true, y_pred_proba, multi_class='ovr', average=average
                )
        except Exception as e:
            logger.warning(f"Failed to compute roc_auc: {e}")
            metrics['roc_auc'] = np.nan
    
    # Matthews correlation coefficient (works for binary and multi-class)
    try:
        metrics['matthews_cc'] = matthews_corrcoef(y_true, y_pred)
    except Exception as e:
        logger.warning(f"Failed to compute matthews_cc: {e}")
        metrics['matthews_cc'] = np.nan
    
    return metrics


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        Confusion matrix (rows=true, columns=predicted)
    """
    return confusion_matrix(y_true, y_pred)


def compute_fold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray = None,
    fold_id: int = None,
    repetition_id: int = None,
    average: str = 'weighted'
) -> Dict[str, Union[float, int]]:
    """
    Compute metrics for a single fold with metadata.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        fold_id: Fold identifier (0-4 for 5-fold CV)
        repetition_id: Repetition identifier (0-2 for 3 repetitions)
        average: Averaging method
    
    Returns:
        Dictionary with fold metrics and metadata
    """
    metrics = compute_classification_metrics(y_true, y_pred, y_pred_proba, average)
    
    # Add metadata
    metrics['n_samples'] = len(y_true)
    metrics['n_classes'] = len(np.unique(y_true))
    
    if fold_id is not None:
        metrics['fold_id'] = fold_id
    
    if repetition_id is not None:
        metrics['repetition_id'] = repetition_id
    
    return metrics


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[int, Dict[str, float]]:
    """
    Compute per-class metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        Dictionary mapping class label to its metrics
    
    Metrics per class:
        - precision: Class-specific precision
        - recall: Class-specific recall
        - f1: Class-specific F1-score
        - support: Number of samples for this class
    """
    classes = np.unique(y_true)
    per_class = {}
    
    for cls in classes:
        y_true_binary = (y_true == cls).astype(int)
        y_pred_binary = (y_pred == cls).astype(int)
        
        per_class[int(cls)] = {
            'precision': precision_score(
                y_true_binary, y_pred_binary, zero_division=0
            ),
            'recall': recall_score(
                y_true_binary, y_pred_binary, zero_division=0
            ),
            'f1': f1_score(
                y_true_binary, y_pred_binary, zero_division=0
            ),
            'support': np.sum(y_true == cls),
        }
    
    return per_class
