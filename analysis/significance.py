"""Significance tests and post-hoc analysis for benchmark results."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import sqrt
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from analysis.statistics import friedman_test, rank_models, validate_score_matrix, wilcoxon_signed_rank


def nemenyi_q_alpha(k: int, alpha: float = 0.05) -> float:
    """Critical q for all-pairs Nemenyi after Friedman.

    Demsar (2006) defines the tabulated Nemenyi critical values as upper
    quantiles of the Studentized range distribution divided by sqrt(2).
    """
    if k < 2:
        raise ValueError('Nemenyi critical values require at least two models')
    if not 0 < alpha < 1:
        raise ValueError('alpha must be between 0 and 1')
    return float(stats.studentized_range.ppf(1.0 - alpha, k, np.inf) / sqrt(2.0))


def nemenyi_critical_difference(k: int, n: int, alpha: float = 0.05) -> float:
    if k < 2 or n < 1:
        raise ValueError('Nemenyi test requires at least two models and one dataset/fold')
    q_alpha = nemenyi_q_alpha(k, alpha=alpha)
    return float(q_alpha * sqrt(k * (k + 1) / (6.0 * n)))


def nemenyi_test(score_matrix: np.ndarray, model_names: Optional[Sequence[str]] = None, alpha: float = 0.05, higher_is_better: bool = True) -> Dict[str, Any]:
    matrix = validate_score_matrix(score_matrix, model_names, procedure='Nemenyi test')

    names = list(model_names) if model_names is not None else [f'model_{i}' for i in range(matrix.shape[1])]
    ranks_per_dataset = np.array([
        stats.rankdata(-row if higher_is_better else row, method='average')
        for row in matrix
    ], dtype=float)
    avg_ranks = ranks_per_dataset.mean(axis=0)
    standard_error = sqrt(matrix.shape[1] * (matrix.shape[1] + 1) / (6.0 * matrix.shape[0]))
    cd = nemenyi_critical_difference(matrix.shape[1], matrix.shape[0], alpha=alpha)

    significance_matrix = pd.DataFrame(False, index=names, columns=names)
    p_value_matrix = pd.DataFrame(1.0, index=names, columns=names)
    for i, j in combinations(range(len(names)), 2):
        rank_diff = abs(avg_ranks[i] - avg_ranks[j])
        statistic = rank_diff / standard_error * sqrt(2.0)
        p_value = float(stats.studentized_range.sf(statistic, matrix.shape[1], np.inf))
        significant = p_value <= alpha
        significance_matrix.iloc[i, j] = significant
        significance_matrix.iloc[j, i] = significant
        p_value_matrix.iloc[i, j] = p_value
        p_value_matrix.iloc[j, i] = p_value

    ranking = pd.DataFrame({'model': names, 'average_rank': avg_ranks}).sort_values(['average_rank', 'model']).reset_index(drop=True)
    ranking['significant_vs_best'] = [
        False,
        *[
            bool(p_value_matrix.loc[ranking.loc[0, 'model'], ranking.loc[idx, 'model']] <= alpha)
            for idx in range(1, len(ranking))
        ],
    ]

    return {
        'critical_difference': cd,
        'q_alpha': nemenyi_q_alpha(matrix.shape[1], alpha=alpha),
        'average_ranks': dict(zip(names, map(float, avg_ranks))),
        'p_value_matrix': p_value_matrix,
        'significance_matrix': significance_matrix,
        'ranking_table': ranking,
        'alpha': alpha,
    }


def pairwise_wilcoxon(score_matrix: np.ndarray, model_names: Optional[Sequence[str]] = None, alternative: str = 'two-sided') -> pd.DataFrame:
    matrix = validate_score_matrix(score_matrix, model_names, procedure='Pairwise Wilcoxon comparison')

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
    pairwise = pd.DataFrame(rows)
    if not pairwise.empty:
        adjusted, reject = holm_bonferroni(pairwise['p_value'].to_numpy(dtype=float), alpha=0.05)
        pairwise['p_value_raw'] = pairwise['p_value']
        pairwise['p_value_adj'] = adjusted
        pairwise['significant'] = reject
        pairwise['decision'] = ['reject null' if flag else 'fail to reject null' for flag in reject]
    return pairwise


def holm_bonferroni(p_values: Iterable[float], alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Apply Holm-Bonferroni correction.

    Args:
        p_values: iterable of raw p-values
        alpha: family-wise error rate

    Returns:
        adjusted_p: array of Holm-adjusted p-values
        reject: boolean array indicating rejection after correction
    """
    pvals = np.asarray(list(p_values), dtype=float)
    m = len(pvals)
    if m == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=bool)

    order = np.argsort(pvals)
    ordered = pvals[order]
    adj = np.empty(m, dtype=float)
    running_max = 0.0
    for idx, p in enumerate(ordered):
        factor = m - idx
        adj_val = min(1.0, p * factor)
        running_max = max(running_max, adj_val)
        adj[idx] = running_max

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adj

    # Holm step-down rejection
    reject = np.zeros(m, dtype=bool)
    for idx, p in enumerate(ordered):
        threshold = alpha / (m - idx)
        if p <= threshold:
            reject[idx] = True
        else:
            break
    reject_unsorted = np.zeros(m, dtype=bool)
    reject_unsorted[order] = reject

    return adjusted, reject_unsorted


def global_significance_analysis(score_matrix: np.ndarray, model_names: Optional[Sequence[str]] = None, higher_is_better: bool = True) -> Dict[str, Any]:
    matrix = validate_score_matrix(score_matrix, model_names, procedure='Global significance analysis', min_observations=2)
    friedman = friedman_test(matrix, model_names=model_names)
    nemenyi = nemenyi_test(matrix, model_names=model_names, higher_is_better=higher_is_better)
    pairwise = pairwise_wilcoxon(matrix, model_names=model_names)
    rankings = rank_models(matrix, model_names=model_names, higher_is_better=higher_is_better)
    return {
        'friedman': friedman,
        'nemenyi': nemenyi,
        'pairwise_wilcoxon': pairwise,
        'ranking_table': rankings,
    }
