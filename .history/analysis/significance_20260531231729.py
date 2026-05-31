"""Significance tests and post-hoc analysis for benchmark results."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import sqrt
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from analysis.statistics import friedman_test, wilcoxon_signed_rank, rank_models


_NEMENYI_Q_ALPHA = {
    0.10: 1.960,
    0.05: 2.241,
    0.01: 2.576,
}


def nemenyi_critical_difference(k: int, n: int, alpha: float = 0.05) -> float:
    if k < 2 or n < 1:
        raise ValueError('Nemenyi test requires at least two models and one dataset/fold')
    q_alpha = _NEMENYI_Q_ALPHA.get(alpha, _NEMENYI_Q_ALPHA[0.05])
    return float(q_alpha * sqrt(k * (k + 1) / (6.0 * n)))


def nemenyi_test(score_matrix: np.ndarray, model_names: Optional[Sequence[str]] = None, alpha: float = 0.05, higher_is_better: bool = True) -> Dict[str, Any]:
    matrix = np.asarray(score_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError('Nemenyi test requires a 2D matrix with at least two models')

    names = list(model_names) if model_names is not None else [f'model_{i}' for i in range(matrix.shape[1])]
    ranks_per_dataset = np.array([
        stats.rankdata(-row if higher_is_better else row, method='average')
        for row in matrix
    ], dtype=float)
    avg_ranks = ranks_per_dataset.mean(axis=0)
    cd = nemenyi_critical_difference(matrix.shape[1], matrix.shape[0], alpha=alpha)

    significance_matrix = pd.DataFrame(False, index=names, columns=names)
    for i, j in combinations(range(len(names)), 2):
        significant = abs(avg_ranks[i] - avg_ranks[j]) > cd
        significance_matrix.iloc[i, j] = significant
        significance_matrix.iloc[j, i] = significant

    ranking = pd.DataFrame({'model': names, 'average_rank': avg_ranks}).sort_values(['average_rank', 'model']).reset_index(drop=True)
    ranking['significant_vs_best'] = [False] + [bool(abs(ranking.loc[idx, 'average_rank'] - ranking.loc[0, 'average_rank']) > cd) for idx in range(1, len(ranking))]

    return {
        'critical_difference': cd,
        'average_ranks': dict(zip(names, map(float, avg_ranks))),
        'significance_matrix': significance_matrix,
        'ranking_table': ranking,
        'alpha': alpha,
    }


def pairwise_wilcoxon(score_matrix: np.ndarray, model_names: Optional[Sequence[str]] = None, alternative: str = 'two-sided') -> pd.DataFrame:
    matrix = np.asarray(score_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError('Pairwise comparison requires a 2D matrix with at least two models')

    names = list(model_names) if model_names is not None else [f'model_{i}' for i in range(matrix.shape[1])]
    rows = []
    for i, j in combinations(range(matrix.shape[1]), 2):
        result = wilcoxon_signed_rank(matrix[:, i], matrix[:, j], alternative=alternative)
        rows.append({
            'model_a': names[i],
            'model_b': names[j],
            'statistic': result['statistic'],
            'p_value': result['p_value'],
            'significant': result['significant'],
            'decision': result['decision'],
        })
    return pd.DataFrame(rows)


def global_significance_analysis(score_matrix: np.ndarray, model_names: Optional[Sequence[str]] = None, higher_is_better: bool = True) -> Dict[str, Any]:
    friedman = friedman_test(score_matrix, model_names=model_names)
    nemenyi = nemenyi_test(score_matrix, model_names=model_names, higher_is_better=higher_is_better)
    pairwise = pairwise_wilcoxon(score_matrix, model_names=model_names)
    rankings = rank_models(score_matrix, model_names=model_names, higher_is_better=higher_is_better)
    return {
        'friedman': friedman,
        'nemenyi': nemenyi,
        'pairwise_wilcoxon': pairwise,
        'ranking_table': rankings,
    }
