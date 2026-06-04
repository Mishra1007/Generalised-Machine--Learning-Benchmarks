"""Statistical tests and uncertainty quantification for benchmark comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import sqrt
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def _as_array(values: Iterable[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=float)


def validate_score_matrix(
    score_matrix: np.ndarray,
    model_names: Optional[Sequence[str]] = None,
    *,
    procedure: str,
    min_observations: int = 1,
    min_models: int = 2,
) -> np.ndarray:
    """Validate benchmark score matrix orientation.

    Contract: rows are aligned observations (folds/repetition-folds/datasets)
    and columns are models. This is the orientation required by Friedman,
    Nemenyi, pairwise Wilcoxon, ranking, and per-model confidence intervals.
    """
    matrix = np.asarray(score_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f'{procedure} requires a 2D score_matrix with rows=observations and columns=models')
    if matrix.shape[0] < min_observations:
        raise ValueError(f'{procedure} requires at least {min_observations} aligned observations')
    if matrix.shape[1] < min_models:
        raise ValueError(f'{procedure} requires at least {min_models} model columns')
    if model_names is not None and len(model_names) != matrix.shape[1]:
        raise ValueError('Model names must match the number of columns in score_matrix')
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f'{procedure} requires finite numeric scores')
    return matrix


def validate_paired_observations(x: Iterable[float], y: Iterable[float], *, procedure: str) -> Tuple[np.ndarray, np.ndarray]:
    """Validate paired vectors aligned by identical observation order."""
    arr_x = _as_array(x)
    arr_y = _as_array(y)
    if arr_x.ndim != 1 or arr_y.ndim != 1:
        raise ValueError(f'{procedure} requires 1D paired observation vectors')
    if len(arr_x) != len(arr_y):
        raise ValueError(f'{procedure} requires paired samples of equal length')
    if len(arr_x) < 1:
        raise ValueError(f'{procedure} requires at least one paired observation')
    if not np.all(np.isfinite(arr_x)) or not np.all(np.isfinite(arr_y)):
        raise ValueError(f'{procedure} requires finite numeric paired observations')
    return arr_x, arr_y


def friedman_test(score_matrix: np.ndarray, model_names: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    matrix = validate_score_matrix(score_matrix, model_names, procedure='Friedman test', min_observations=2)

    columns = [matrix[:, idx] for idx in range(matrix.shape[1])]
    statistic, p_value = stats.friedmanchisquare(*columns)
    interpretation = 'significant differences detected' if p_value < 0.05 else 'no significant differences detected'
    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'interpretation': interpretation,
        'model_names': list(model_names) if model_names is not None else [f'model_{i}' for i in range(matrix.shape[1])],
    }


def wilcoxon_signed_rank(x: Iterable[float], y: Iterable[float], zero_method: str = 'wilcox', alternative: str = 'two-sided') -> Dict[str, Any]:
    arr_x, arr_y = validate_paired_observations(x, y, procedure='Wilcoxon test')

    statistic, p_value = stats.wilcoxon(arr_x, arr_y, zero_method=zero_method, alternative=alternative, mode='auto')
    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': bool(p_value < 0.05),
        'decision': 'reject null' if p_value < 0.05 else 'fail to reject null',
    }


def mean_confidence_interval(values: Iterable[float], confidence: float = 0.95) -> Dict[str, float]:
    arr = _as_array(values)
    if len(arr) < 2:
        mean = float(arr.mean()) if len(arr) else float('nan')
        return {'mean': mean, 'lower': mean, 'upper': mean, 'confidence': confidence}

    mean = float(arr.mean())
    sem = stats.sem(arr, nan_policy='omit')
    if np.isnan(sem):
        return {'mean': mean, 'lower': mean, 'upper': mean, 'confidence': confidence}
    t_crit = stats.t.ppf((1 + confidence) / 2.0, df=len(arr) - 1)
    margin = float(t_crit * sem)
    return {'mean': mean, 'lower': mean - margin, 'upper': mean + margin, 'confidence': confidence}


def bootstrap_confidence_interval(values: Iterable[float], confidence: float = 0.95, n_bootstrap: int = 10000, random_state: Optional[int] = 42) -> Dict[str, float]:
    arr = _as_array(values)
    if len(arr) == 0:
        return {'mean': float('nan'), 'lower': float('nan'), 'upper': float('nan'), 'confidence': confidence}

    rng = np.random.default_rng(random_state)
    boot_means = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_bootstrap)], dtype=float)
    alpha = (1 - confidence) / 2.0
    lower = float(np.quantile(boot_means, alpha))
    upper = float(np.quantile(boot_means, 1 - alpha))
    return {'mean': float(arr.mean()), 'lower': lower, 'upper': upper, 'confidence': confidence}


def rank_models(score_matrix: np.ndarray, model_names: Optional[Sequence[str]] = None, higher_is_better: bool = True) -> pd.DataFrame:
    matrix = validate_score_matrix(score_matrix, model_names, procedure='Model ranking')

    names = list(model_names) if model_names is not None else [f'model_{i}' for i in range(matrix.shape[1])]
    means = matrix.mean(axis=0)
    ranks = stats.rankdata(-means if higher_is_better else means, method='average')
    table = pd.DataFrame({'model': names, 'mean_score': means, 'rank': ranks})
    table = table.sort_values(['rank', 'model']).reset_index(drop=True)
    table['rank'] = table['rank'].astype(float)
    return table
