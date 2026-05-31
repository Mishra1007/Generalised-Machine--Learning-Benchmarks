"""Normalization utilities for metrics (per-dataset min-max).

Normalization rules:
- Per-dataset Min-Max normalization to [0,1]
- Higher values always better after normalization
- Computational time metrics are inverted via 1/(1+time) before normalization
- Stability variance metrics are inverted via 1/(1+cv) before normalization

The main entry point is `normalize_model_summaries(models_summaries, primary_metric='f1')`
which returns normalized scores for the metrics used in CBS.
"""

from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


def _safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for p in path:
        if cur is None:
            return default
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur


def _invert_time(t: float) -> float:
    """Invert time so that higher is better.

    Uses 1/(1+time) mapping which keeps monotonicity and bounds values to (0,1].
    """
    try:
        return 1.0 / (1.0 + float(t))
    except Exception:
        return 0.0


def _invert_cv(cv: float) -> float:
    """Invert coefficient of variation (lower cv is better).

    Uses 1/(1+cv).
    """
    try:
        return 1.0 / (1.0 + float(cv))
    except Exception:
        return 0.0


def _min_max_normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    min_v = min(values)
    max_v = max(values)
    if max_v == min_v:
        # All identical: return all ones to avoid unfair penalization
        return [1.0 for _ in values]
    denom = max_v - min_v
    return [(v - min_v) / denom for v in values]


def extract_metrics_for_cbs(summary: Dict[str, Any], primary_metric: str = 'f1') -> Dict[str, float]:
    """Extract raw metric values needed for CBS from a model summary.

    Tries to gather (raw): f1, roc_auc, accuracy, precision, recall, total_time, stability_score or cv

    Returns a dict with keys: 'f1','roc_auc','accuracy','precision','recall','time','stability_cv','stability_score'
    Missing numeric values are treated as 0.0
    """
    out = {}
    overall = summary.get('overall', {}) if isinstance(summary, dict) else {}

    def get_overall(name):
        return overall.get(f"{name}_mean") if overall.get(f"{name}_mean") is not None else overall.get(name)

    out['f1'] = float(get_overall('f1') or 0.0)
    out['roc_auc'] = float(get_overall('roc_auc') or 0.0)
    out['accuracy'] = float(get_overall('accuracy') or 0.0)
    out['precision'] = float(get_overall('precision') or 0.0)
    out['recall'] = float(get_overall('recall') or 0.0)

    # Time: prefer total_time_mean, fallback to train+eval
    total_time = overall.get('total_time_mean')
    if total_time is None:
        train_t = overall.get('train_time_mean') or 0.0
        eval_t = overall.get('eval_time_mean') or 0.0
        total_time = train_t + eval_t
    out['time'] = float(total_time or 0.0)

    # Stability: try stability.by_metric[primary_metric].stability_score
    stability_score = _safe_get(summary, ['stability', 'by_metric', primary_metric, 'stability_score'], None)
    if stability_score is not None:
        out['stability_score'] = float(stability_score)
        out['stability_cv'] = None
    else:
        # Fallback to assessment primary_metric_cv
        cv = _safe_get(summary, ['stability', 'assessment', 'primary_metric_cv'], None)
        if cv is not None:
            out['stability_cv'] = float(cv)
            out['stability_score'] = None
        else:
            out['stability_cv'] = None
            out['stability_score'] = None

    return out


def normalize_model_summaries(models_summaries: Dict[str, Dict[str, Any]], primary_metric: str = 'f1') -> Dict[str, Dict[str, float]]:
    """Normalize a collection of model summaries per dataset using per-dataset Min-Max normalization.

    Args:
        models_summaries: Mapping from model_name -> summary dict (as produced by MetricsCalculator.get_comprehensive_summary)
        primary_metric: Primary metric used for stability lookup (default 'f1')

    Returns:
        Mapping model_name -> normalized metrics dict with keys:
          'f1','roc_auc','accuracy','precision','recall','time_score','stability_score'
        Each value is in [0,1] with higher being better.
    """
    # Extract raw metrics
    raw_map = {}
    for name, summary in models_summaries.items():
        raw_map[name] = extract_metrics_for_cbs(summary, primary_metric=primary_metric)

    # Prepare lists for normalization
    f1_vals = [raw_map[m]['f1'] for m in raw_map]
    roc_vals = [raw_map[m]['roc_auc'] for m in raw_map]
    acc_vals = [raw_map[m]['accuracy'] for m in raw_map]
    prec_vals = [raw_map[m]['precision'] for m in raw_map]
    rec_vals = [raw_map[m]['recall'] for m in raw_map]

    # For time and stability we apply inversion first
    time_inverted = [_invert_time(raw_map[m]['time']) for m in raw_map]

    # For stability: prefer stability_score else invert cv
    stability_raw = []
    for m in raw_map:
        if raw_map[m].get('stability_score') is not None:
            stability_raw.append(float(raw_map[m]['stability_score']))
        elif raw_map[m].get('stability_cv') is not None:
            stability_raw.append(_invert_cv(raw_map[m]['stability_cv']))
        else:
            stability_raw.append(0.0)

    # Normalize all
    f1_n = _min_max_normalize(f1_vals)
    roc_n = _min_max_normalize(roc_vals)
    acc_n = _min_max_normalize(acc_vals)
    prec_n = _min_max_normalize(prec_vals)
    rec_n = _min_max_normalize(rec_vals)
    time_n = _min_max_normalize(time_inverted)
    stability_n = _min_max_normalize(stability_raw)

    # Pack results
    normalized = {}
    names = list(raw_map.keys())
    for i, name in enumerate(names):
        normalized[name] = {
            'f1': float(f1_n[i]) if f1_n else 0.0,
            'roc_auc': float(roc_n[i]) if roc_n else 0.0,
            'accuracy': float(acc_n[i]) if acc_n else 0.0,
            'precision': float(prec_n[i]) if prec_n else 0.0,
            'recall': float(rec_n[i]) if rec_n else 0.0,
            'time_score': float(time_n[i]) if time_n else 0.0,
            'stability_score': float(stability_n[i]) if stability_n else 0.0,
        }

    return normalized
