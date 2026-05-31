"""CBS (Composite Benchmark Score) calculator and ranking utilities.

CBS Formula (weights applied on normalized metrics):
CBS = 0.30 * F1
    + 0.20 * ROC-AUC
    + 0.10 * Accuracy
    + 0.10 * Precision
    + 0.10 * Recall
    + 0.10 * Time Score
    + 0.10 * Stability Score

All input metrics must be normalized to [0,1] with higher better (use metrics.normalization.normalize_model_summaries()).
"""
from typing import Dict, Any, List, Tuple
import logging

from metrics.normalization import normalize_model_summaries

logger = logging.getLogger(__name__)

CBS_WEIGHTS = {
    'f1': 0.30,
    'roc_auc': 0.20,
    'accuracy': 0.10,
    'precision': 0.10,
    'recall': 0.10,
    'time_score': 0.10,
    'stability_score': 0.10,
}


def compute_cbs_from_normalized(normalized_map: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Compute CBS for each model given normalized metric map.

    Args:
        normalized_map: model -> {metric: normalized_value}

    Returns:
        model -> cbs_score (float)
    """
    results = {}
    for model_name, metrics in normalized_map.items():
        score = 0.0
        for m, w in CBS_WEIGHTS.items():
            score += float(metrics.get(m, 0.0)) * w
        results[model_name] = float(score)
    return results


def compute_cbs(models_summaries: Dict[str, Dict[str, Any]], primary_metric: str = 'f1') -> Dict[str, Any]:
    """Convenience wrapper: normalize then compute CBS and return combined info.

    Returns mapping model_name -> {'normalized': {...}, 'cbs': float}
    """
    normalized = normalize_model_summaries(models_summaries, primary_metric=primary_metric)
    cbs_map = compute_cbs_from_normalized(normalized)

    out = {}
    for m in normalized:
        out[m] = {
            'normalized': normalized[m],
            'cbs': cbs_map.get(m, 0.0),
        }
    return out


def rank_models_by_cbs(cbs_map: Dict[str, Any]) -> List[Tuple[str, float]]:
    """Return list of (model_name, cbs) sorted descending by CBS."""
    items = [(name, info['cbs'] if isinstance(info, dict) else float(info)) for name, info in cbs_map.items()]
    items_sorted = sorted(items, key=lambda x: x[1], reverse=True)
    return items_sorted


def top_k_models(cbs_map: Dict[str, Any], k: int = 3) -> List[Tuple[str, float]]:
    ranked = rank_models_by_cbs(cbs_map)
    return ranked[:k]
