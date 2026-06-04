"""Scientific CBS sensitivity helpers.

These functions analyze the existing CBS definition without changing CBS
weights or benchmark execution behavior.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from metrics.cbs import CBS_WEIGHTS


CBS_METRICS = [
    'f1',
    'roc_auc',
    'accuracy',
    'precision',
    'recall',
    'time_score',
    'stability_score',
]


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError('CBS weights must sum to a positive value')
    return {metric: float(weight) / total for metric, weight in weights.items()}


def compute_weighted_scores(frame: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    normalized = normalize_weights(weights)
    missing = [metric for metric in normalized if metric not in frame.columns]
    if missing:
        raise ValueError(f'Missing CBS metric columns: {missing}')
    score = pd.Series(0.0, index=frame.index, dtype=float)
    for metric, weight in normalized.items():
        score = score + frame[metric].astype(float) * weight
    return score.astype(float)


def rank_scores(scores: pd.Series) -> pd.Series:
    return scores.rank(ascending=False, method='average').astype(float)


def ranking_reversal_count(baseline_rank: pd.Series, perturbed_rank: pd.Series) -> int:
    reversals = 0
    common = list(baseline_rank.index.intersection(perturbed_rank.index))
    for left, right in combinations(common, 2):
        base_order = np.sign(baseline_rank.loc[left] - baseline_rank.loc[right])
        pert_order = np.sign(perturbed_rank.loc[left] - perturbed_rank.loc[right])
        if base_order != 0 and pert_order != 0 and base_order != pert_order:
            reversals += 1
    return int(reversals)


def rank_correlation_summary(baseline_rank: pd.Series, perturbed_rank: pd.Series) -> Tuple[float, float]:
    if len(baseline_rank) < 2:
        return float('nan'), float('nan')
    aligned = pd.concat([baseline_rank.rename('baseline'), perturbed_rank.rename('perturbed')], axis=1).dropna()
    if aligned['baseline'].nunique() <= 1 or aligned['perturbed'].nunique() <= 1:
        return float('nan'), float('nan')
    spearman = stats.spearmanr(aligned['baseline'], aligned['perturbed']).correlation
    kendall = stats.kendalltau(aligned['baseline'], aligned['perturbed']).correlation
    return float(spearman), float(kendall)


def perturb_weight_grid(
    frame: pd.DataFrame,
    perturbation_levels: Sequence[float] = (0.05, 0.10, 0.20),
    weights: Dict[str, float] = CBS_WEIGHTS,
) -> pd.DataFrame:
    base_weights = normalize_weights(weights)
    baseline_scores = compute_weighted_scores(frame, base_weights)
    baseline_rank = rank_scores(baseline_scores)
    model_names = frame['model'] if 'model' in frame.columns else frame.index.astype(str)
    baseline_rank.index = model_names
    baseline_scores.index = model_names

    rows: List[Dict[str, Any]] = []
    for level in perturbation_levels:
        for metric in CBS_METRICS:
            for direction in (-1.0, 1.0):
                perturbed_weights = dict(base_weights)
                perturbed_weights[metric] = max(0.0, perturbed_weights[metric] * (1.0 + direction * level))
                perturbed_weights = normalize_weights(perturbed_weights)
                perturbed_scores = compute_weighted_scores(frame, perturbed_weights)
                perturbed_scores.index = model_names
                perturbed_rank = rank_scores(perturbed_scores)
                spearman, kendall = rank_correlation_summary(baseline_rank, perturbed_rank)
                reversals = ranking_reversal_count(baseline_rank, perturbed_rank)

                for model in model_names:
                    rows.append({
                        'model': model,
                        'metric': metric,
                        'perturbation_level': float(level),
                        'direction': 'increase' if direction > 0 else 'decrease',
                        'baseline_weight': base_weights[metric],
                        'perturbed_weight': perturbed_weights[metric],
                        'baseline_cbs': baseline_scores.loc[model],
                        'perturbed_cbs': perturbed_scores.loc[model],
                        'score_delta': perturbed_scores.loc[model] - baseline_scores.loc[model],
                        'baseline_rank': baseline_rank.loc[model],
                        'perturbed_rank': perturbed_rank.loc[model],
                        'rank_shift': perturbed_rank.loc[model] - baseline_rank.loc[model],
                        'spearman_rank_correlation': spearman,
                        'kendall_rank_correlation': kendall,
                        'ranking_reversals': reversals,
                    })
    return pd.DataFrame(rows)


def random_weight_vectors(
    n_samples: int,
    metrics: Sequence[str] = CBS_METRICS,
    random_state: int = 42,
    concentration: float = 20.0,
) -> pd.DataFrame:
    base = np.asarray([CBS_WEIGHTS[metric] for metric in metrics], dtype=float)
    alpha = np.maximum(base * concentration, 1e-6)
    rng = np.random.default_rng(random_state)
    samples = rng.dirichlet(alpha, size=n_samples)
    return pd.DataFrame(samples, columns=list(metrics))


def monte_carlo_weight_analysis(
    frame: pd.DataFrame,
    n_samples: int = 5000,
    random_state: int = 42,
    concentration: float = 20.0,
    weights: Dict[str, float] = CBS_WEIGHTS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    model_names = frame['model'] if 'model' in frame.columns else frame.index.astype(str)
    base_scores = compute_weighted_scores(frame, weights)
    base_scores.index = model_names
    base_rank = rank_scores(base_scores)
    weight_samples = random_weight_vectors(n_samples, random_state=random_state, concentration=concentration)

    rank_records: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    top_counts = {model: 0 for model in model_names}
    rank_values = {model: [] for model in model_names}
    score_values = {model: [] for model in model_names}
    spearman_values: List[float] = []
    kendall_values: List[float] = []
    reversal_values: List[int] = []

    for sample_id, weight_row in weight_samples.iterrows():
        sample_weights = weight_row.to_dict()
        scores = compute_weighted_scores(frame, sample_weights)
        scores.index = model_names
        ranks = rank_scores(scores)
        ordered = scores.sort_values(ascending=False)
        top_counts[ordered.index[0]] += 1
        spearman, kendall = rank_correlation_summary(base_rank, ranks)
        reversals = ranking_reversal_count(base_rank, ranks)
        if not np.isnan(spearman):
            spearman_values.append(spearman)
        if not np.isnan(kendall):
            kendall_values.append(kendall)
        reversal_values.append(reversals)

        for model in model_names:
            rank_values[model].append(float(ranks.loc[model]))
            score_values[model].append(float(scores.loc[model]))
            rank_records.append({
                'sample_id': int(sample_id),
                'model': model,
                'sampled_cbs': float(scores.loc[model]),
                'sampled_rank': float(ranks.loc[model]),
                'baseline_rank': float(base_rank.loc[model]),
                'rank_shift': float(ranks.loc[model] - base_rank.loc[model]),
                'top_model': bool(model == ordered.index[0]),
            })

    for model in model_names:
        ranks = np.asarray(rank_values[model], dtype=float)
        scores = np.asarray(score_values[model], dtype=float)
        summary_rows.append({
            'model': model,
            'baseline_cbs': float(base_scores.loc[model]),
            'baseline_rank': float(base_rank.loc[model]),
            'mean_sampled_cbs': float(scores.mean()),
            'std_sampled_cbs': float(scores.std(ddof=1)),
            'mean_rank': float(ranks.mean()),
            'rank_std': float(ranks.std(ddof=1)),
            'top_model_frequency': float(top_counts[model] / n_samples),
            'rank_1_frequency': float(np.mean(ranks == 1.0)),
            'rank_reversal_frequency': float(np.mean(ranks != base_rank.loc[model])),
            'mean_spearman_vs_baseline': float(np.mean(spearman_values)) if spearman_values else np.nan,
            'mean_kendall_vs_baseline': float(np.mean(kendall_values)) if kendall_values else np.nan,
            'mean_pairwise_reversals': float(np.mean(reversal_values)) if reversal_values else np.nan,
        })

    return pd.DataFrame(summary_rows), pd.DataFrame(rank_records)
