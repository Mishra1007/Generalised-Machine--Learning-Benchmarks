"""CBS validation and robustness analysis utilities.

This module consumes saved benchmark artifacts and validates the behavior of the
existing Composite Benchmark Score (CBS) without modifying the CBS definition.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from analysis.cbs_reports import write_cbs_validation_report
from analysis.cbs_sensitivity import (
    monte_carlo_weight_analysis,
    perturb_weight_grid,
)
from metrics.cbs import CBS_WEIGHTS

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - plotting is optional in some envs
    plt = None

try:
    import seaborn as sns
except Exception:  # pragma: no cover - optional dependency
    sns = None


CBS_METRICS = [
    'f1',
    'roc_auc',
    'accuracy',
    'precision',
    'recall',
    'time_score',
    'stability_score',
]


@dataclass
class CBSValidationArtifacts:
    report_path: str
    sensitivity_path: str
    weight_robustness_path: str
    metric_influence_path: str
    ranking_stability_path: str
    monte_carlo_samples_path: str
    correlation_path: str
    normalization_path: str
    plots: Dict[str, Tuple[str, str]]


def _require_matplotlib() -> None:
    if plt is None:
        raise RuntimeError('matplotlib is required to generate CBS validation plots')


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str) and not value.strip():
            return default
        return float(value)
    except Exception:
        return default


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError('CBS weights must sum to a positive value')
    return {k: float(v) / total for k, v in weights.items()}


def _rank_models(scores: pd.Series, higher_is_better: bool = True) -> pd.Series:
    order = scores.rank(ascending=not higher_is_better, method='average')
    return order.astype(float)


def _compute_cbs_row(row: pd.Series, weights: Dict[str, float]) -> float:
    return float(sum(_safe_float(row.get(metric, 0.0)) * weight for metric, weight in weights.items()))


def load_results_bundle(results_dir: str | Path) -> Dict[str, pd.DataFrame]:
    base = Path(results_dir)
    normalized_path = base / 'normalized_results.csv'
    cbs_path = base / 'cbs_scores.csv'
    raw_path = base / 'raw_results.csv'

    if not normalized_path.exists():
        raise FileNotFoundError(f'Missing normalized results: {normalized_path}')
    if not cbs_path.exists():
        raise FileNotFoundError(f'Missing CBS scores: {cbs_path}')

    normalized = pd.read_csv(normalized_path)
    cbs_scores = pd.read_csv(cbs_path)
    raw = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()

    return {
        'normalized': normalized,
        'cbs': cbs_scores,
        'raw': raw,
    }


def build_cbs_frame(normalized: pd.DataFrame, weights: Dict[str, float] = CBS_WEIGHTS) -> pd.DataFrame:
    weights = _normalize_weights(weights)
    df = normalized.copy()
    missing = [metric for metric in CBS_METRICS if metric not in df.columns]
    if missing:
        raise ValueError(f'Missing CBS component columns: {missing}')

    df['cbs'] = df.apply(lambda row: _compute_cbs_row(row, weights), axis=1)
    for metric, weight in weights.items():
        df[f'contribution__{metric}'] = df[metric] * weight
        df[f'contribution_pct__{metric}'] = np.where(df['cbs'] > 0, df[f'contribution__{metric}'] / df['cbs'], np.nan)

    df['baseline_rank'] = _rank_models(df['cbs'], higher_is_better=True)
    return df.sort_values(['cbs', 'model'], ascending=[False, True]).reset_index(drop=True)


def _apply_weight_perturbation(weights: Dict[str, float], metric: str, pct: float) -> Dict[str, float]:
    perturbed = dict(weights)
    perturbed[metric] = perturbed[metric] * (1.0 + pct)
    return _normalize_weights(perturbed)


def _apply_metric_perturbation(row: pd.Series, metric: str, delta: float) -> float:
    return float(np.clip(_safe_float(row.get(metric, 0.0)) + delta, 0.0, 1.0))


def sensitivity_analysis(cbs_frame: pd.DataFrame, perturbation: float = 0.01, noise_std: float = 0.02, fold_std: Optional[pd.Series] = None, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_state)
    rows: List[Dict[str, Any]] = []

    for _, row in cbs_frame.iterrows():
        baseline = float(row['cbs'])
        for metric in CBS_METRICS:
            base_value = _safe_float(row.get(metric, 0.0))
            up_value = _apply_metric_perturbation(row, metric, perturbation)
            down_value = _apply_metric_perturbation(row, metric, -perturbation)

            row_up = row.copy()
            row_down = row.copy()
            row_up[metric] = up_value
            row_down[metric] = down_value
            cbs_up = _compute_cbs_row(row_up, CBS_WEIGHTS)
            cbs_down = _compute_cbs_row(row_down, CBS_WEIGHTS)

            rows.append({
                'model': row['model'],
                'metric': metric,
                'scenario': 'small_perturbation',
                'baseline_metric': base_value,
                'perturbation': perturbation,
                'baseline_cbs': baseline,
                'perturbed_cbs': cbs_up,
                'delta_cbs': cbs_up - baseline,
                'absolute_delta_cbs': abs(cbs_up - baseline),
                'direction': 'up',
            })
            rows.append({
                'model': row['model'],
                'metric': metric,
                'scenario': 'small_perturbation',
                'baseline_metric': base_value,
                'perturbation': -perturbation,
                'baseline_cbs': baseline,
                'perturbed_cbs': cbs_down,
                'delta_cbs': cbs_down - baseline,
                'absolute_delta_cbs': abs(cbs_down - baseline),
                'direction': 'down',
            })

            noise = rng.normal(0.0, noise_std, size=128)
            noisy_scores = []
            for sample_noise in noise:
                row_noise = row.copy()
                row_noise[metric] = float(np.clip(base_value + sample_noise, 0.0, 1.0))
                noisy_scores.append(_compute_cbs_row(row_noise, CBS_WEIGHTS))
            rows.append({
                'model': row['model'],
                'metric': metric,
                'scenario': 'noisy_measurement',
                'baseline_metric': base_value,
                'perturbation': noise_std,
                'baseline_cbs': baseline,
                'perturbed_cbs': float(np.mean(noisy_scores)),
                'delta_cbs': float(np.mean(noisy_scores) - baseline),
                'absolute_delta_cbs': float(np.mean(np.abs(np.asarray(noisy_scores) - baseline))),
                'cbs_std': float(np.std(noisy_scores, ddof=1)),
                'cbs_lower': float(np.quantile(noisy_scores, 0.025)),
                'cbs_upper': float(np.quantile(noisy_scores, 0.975)),
                'direction': 'noise',
            })

            variability_std = _safe_float(fold_std.loc[row['model']]) if fold_std is not None and row['model'] in fold_std.index else noise_std
            fold_scores = []
            for sample_noise in rng.normal(0.0, variability_std, size=256):
                row_fold = row.copy()
                row_fold[metric] = float(np.clip(base_value + sample_noise, 0.0, 1.0))
                fold_scores.append(_compute_cbs_row(row_fold, CBS_WEIGHTS))
            rows.append({
                'model': row['model'],
                'metric': metric,
                'scenario': 'fold_variability',
                'baseline_metric': base_value,
                'perturbation': variability_std,
                'baseline_cbs': baseline,
                'perturbed_cbs': float(np.mean(fold_scores)),
                'delta_cbs': float(np.mean(fold_scores) - baseline),
                'absolute_delta_cbs': float(np.mean(np.abs(np.asarray(fold_scores) - baseline))),
                'cbs_std': float(np.std(fold_scores, ddof=1)),
                'cbs_lower': float(np.quantile(fold_scores, 0.025)),
                'cbs_upper': float(np.quantile(fold_scores, 0.975)),
                'direction': 'fold',
            })

    sensitivity = pd.DataFrame(rows)
    if sensitivity.empty:
        ranking = pd.DataFrame(columns=['metric', 'mean_absolute_delta_cbs', 'rank'])
        return sensitivity, ranking

    ranking = (
        sensitivity.groupby(['metric', 'scenario'], as_index=False)['absolute_delta_cbs']
        .mean()
        .rename(columns={'absolute_delta_cbs': 'mean_absolute_delta_cbs'})
    )
    ranking['rank'] = ranking.groupby('scenario')['mean_absolute_delta_cbs'].rank(ascending=False, method='dense')
    ranking = ranking.sort_values(['scenario', 'rank', 'metric']).reset_index(drop=True)
    return sensitivity, ranking


def weight_robustness_analysis(cbs_frame: pd.DataFrame, perturbation_levels: Sequence[float] = (0.05, 0.10, 0.20), weights: Dict[str, float] = CBS_WEIGHTS) -> pd.DataFrame:
    data = perturb_weight_grid(cbs_frame, perturbation_levels=perturbation_levels, weights=weights)
    if not data.empty:
        data['ranking_stability'] = data['spearman_rank_correlation']
    return data


def metric_dominance_analysis(cbs_frame: pd.DataFrame, weights: Dict[str, float] = CBS_WEIGHTS) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    contribution_vars = {}
    total_contribution_variance = 0.0
    for metric in CBS_METRICS:
        variance = float(cbs_frame[f'contribution__{metric}'].var(ddof=1)) if len(cbs_frame) > 1 else 0.0
        contribution_vars[metric] = variance
        total_contribution_variance += max(0.0, variance)

    for metric in CBS_METRICS:
        contribution = cbs_frame[f'contribution__{metric}']
        share = cbs_frame[f'contribution_pct__{metric}']
        loo_scores = cbs_frame.apply(lambda row: _compute_cbs_row(row.drop(labels=[metric]), {k: v for k, v in weights.items() if k != metric}) / max(1e-12, 1.0 - weights[metric]), axis=1)
        variance_contribution_pct = contribution_vars[metric] / total_contribution_variance if total_contribution_variance > 0 else np.nan
        rows.append({
            'metric': metric,
            'weight': weights[metric],
            'mean_contribution': float(contribution.mean()),
            'contribution_variance': contribution_vars[metric],
            'variance_contribution_pct': float(variance_contribution_pct),
            'mean_contribution_pct': float(share.mean()),
            'max_contribution_pct': float(share.max()),
            'min_contribution_pct': float(share.min()),
            'mean_leave_one_out_impact': float((cbs_frame['cbs'] - loo_scores).abs().mean()),
            'pearson_cbs_corr': float(stats.pearsonr(cbs_frame['cbs'], cbs_frame[metric]).statistic if len(cbs_frame) > 1 else np.nan),
            'spearman_cbs_corr': float(stats.spearmanr(cbs_frame['cbs'], cbs_frame[metric]).correlation if len(cbs_frame) > 1 else np.nan),
            'metric_mean': float(cbs_frame[metric].mean()),
            'metric_std': float(cbs_frame[metric].std(ddof=1)),
        })

    dominance = pd.DataFrame(rows)
    dominance['impact_rank'] = dominance['mean_leave_one_out_impact'].rank(ascending=False, method='dense')
    dominance['contribution_rank'] = dominance['mean_contribution_pct'].rank(ascending=False, method='dense')
    return dominance.sort_values(['impact_rank', 'metric']).reset_index(drop=True)


def correlation_analysis(cbs_frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    for metric in CBS_METRICS:
        pearson = stats.pearsonr(cbs_frame['cbs'], cbs_frame[metric]) if len(cbs_frame) > 1 else (np.nan, np.nan)
        spearman = stats.spearmanr(cbs_frame['cbs'], cbs_frame[metric]) if len(cbs_frame) > 1 else (np.nan, np.nan)
        rows.append({
            'target': 'cbs',
            'metric': metric,
            'pearson_r': float(pearson.statistic if hasattr(pearson, 'statistic') else pearson[0]),
            'pearson_p': float(pearson.pvalue if hasattr(pearson, 'pvalue') else pearson[1]),
            'spearman_r': float(spearman.correlation if hasattr(spearman, 'correlation') else spearman[0]),
            'spearman_p': float(spearman.pvalue if hasattr(spearman, 'pvalue') else spearman[1]),
        })

    summary = pd.DataFrame(rows)

    corr_df = cbs_frame[['cbs'] + CBS_METRICS].corr(method='pearson')
    redundant_pairs: List[Dict[str, Any]] = []
    for i, left in enumerate(['cbs'] + CBS_METRICS):
        for right in (['cbs'] + CBS_METRICS)[i + 1:]:
            value = float(corr_df.loc[left, right])
            if abs(value) >= 0.90:
                redundant_pairs.append({'metric_a': left, 'metric_b': right, 'pearson_r': value, 'redundant': True})

    redundancy = pd.DataFrame(redundant_pairs)
    return summary, redundancy


def normalization_validation(raw: pd.DataFrame, normalized: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    raw_map = raw.set_index('model')
    norm_map = normalized.set_index('model')

    raw_columns = {
        'accuracy': 'overall__accuracy_mean',
        'f1': 'overall__f1_mean',
        'precision': 'overall__precision_mean',
        'recall': 'overall__recall_mean',
        'roc_auc': 'overall__roc_auc_mean',
        'time_score': 'overall__total_time_mean',
        'stability_score': 'stability__assessment__primary_metric_cv',
    }

    for metric, raw_col in raw_columns.items():
        if raw_col not in raw_map.columns or metric not in norm_map.columns:
            continue

        raw_values = raw_map[raw_col].astype(float)
        if metric == 'time_score':
            raw_values = 1.0 / (1.0 + raw_values)
        elif metric == 'stability_score':
            raw_values = 1.0 / (1.0 + raw_values)

        norm_values = norm_map[metric].astype(float)
        raw_rank = raw_values.rank(ascending=False, method='average')
        norm_rank = norm_values.rank(ascending=False, method='average')
        spearman = stats.spearmanr(raw_values, norm_values)
        bounded = bool(((norm_values >= 0.0) & (norm_values <= 1.0)).all())
        ranking_distortion = float((raw_rank - norm_rank).abs().sum())
        artifact_reasons = []
        if not bounded:
            artifact_reasons.append('normalized values outside [0,1]')
        if raw_values.nunique() > 1 and norm_values.nunique() <= 1:
            artifact_reasons.append('non-constant raw values collapsed to one normalized value')
        if ranking_distortion > 0:
            artifact_reasons.append('raw-to-normalized rank distortion')

        rows.append({
            'metric': metric,
            'raw_min': float(raw_values.min()),
            'raw_max': float(raw_values.max()),
            'raw_std': float(raw_values.std(ddof=1)),
            'normalized_min': float(norm_values.min()),
            'normalized_max': float(norm_values.max()),
            'normalized_std': float(norm_values.std(ddof=1)),
            'compression_ratio': float(norm_values.std(ddof=1) / raw_values.std(ddof=1)) if raw_values.std(ddof=1) else np.nan,
            'rank_correlation': float(spearman.correlation),
            'rank_p_value': float(spearman.pvalue),
            'ranking_distortion': ranking_distortion,
            'unique_raw_values': int(raw_values.nunique()),
            'unique_normalized_values': int(norm_values.nunique()),
            'bounded_0_1': bounded,
            'artifact_flag': bool(artifact_reasons),
            'artifact_reason': '; '.join(artifact_reasons),
        })

    proxy_weights = _normalize_weights(CBS_WEIGHTS)
    raw_proxy = []
    for model in raw_map.index.intersection(norm_map.index):
        series = raw_map.loc[model]
        proxy = (
            proxy_weights['f1'] * _safe_float(series.get('overall__f1_mean'))
            + proxy_weights['roc_auc'] * _safe_float(series.get('overall__roc_auc_mean'))
            + proxy_weights['accuracy'] * _safe_float(series.get('overall__accuracy_mean'))
            + proxy_weights['precision'] * _safe_float(series.get('overall__precision_mean'))
            + proxy_weights['recall'] * _safe_float(series.get('overall__recall_mean'))
            + proxy_weights['time_score'] * (1.0 / (1.0 + _safe_float(series.get('overall__total_time_mean'))))
            + proxy_weights['stability_score'] * (1.0 / (1.0 + _safe_float(series.get('stability__assessment__primary_metric_cv'))))
        )
        raw_proxy.append((model, proxy))

    if raw_proxy:
        proxy_df = pd.DataFrame(raw_proxy, columns=['model', 'raw_proxy_cbs'])
        merged = norm_map[['cbs']].join(proxy_df.set_index('model'), how='inner')
        if len(merged) > 1:
            norm_values = merged['cbs']
            raw_values = merged['raw_proxy_cbs']
            raw_rank = raw_values.rank(ascending=False)
            norm_rank = norm_values.rank(ascending=False)
            bounded = bool(((norm_values >= 0.0) & (norm_values <= 1.0)).all())
            ranking_distortion = float((raw_rank - norm_rank).abs().sum())
            artifact_reasons = []
            if not bounded:
                artifact_reasons.append('CBS outside [0,1]')
            if raw_values.nunique() > 1 and norm_values.nunique() <= 1:
                artifact_reasons.append('raw proxy values collapsed to one normalized CBS value')
            if ranking_distortion > 0:
                artifact_reasons.append('raw proxy rank differs from normalized CBS rank')
            rows.append({
                'metric': 'cbs_proxy',
                'raw_min': float(raw_values.min()),
                'raw_max': float(raw_values.max()),
                'raw_std': float(raw_values.std(ddof=1)),
                'normalized_min': float(norm_values.min()),
                'normalized_max': float(norm_values.max()),
                'normalized_std': float(norm_values.std(ddof=1)),
                'compression_ratio': float(norm_values.std(ddof=1) / raw_values.std(ddof=1)) if raw_values.std(ddof=1) else np.nan,
                'rank_correlation': float(stats.spearmanr(raw_values, norm_values).correlation),
                'rank_p_value': float(stats.spearmanr(raw_values, norm_values).pvalue),
                'ranking_distortion': ranking_distortion,
                'unique_raw_values': int(raw_values.nunique()),
                'unique_normalized_values': int(norm_values.nunique()),
                'bounded_0_1': bounded,
                'artifact_flag': bool(artifact_reasons),
                'artifact_reason': '; '.join(artifact_reasons),
            })

    return pd.DataFrame(rows)


def monte_carlo_stability(cbs_frame: pd.DataFrame, iterations: int = 5000, noise_std: float = 0.02, random_state: int = 42, fold_std: Optional[pd.Series] = None) -> pd.DataFrame:
    summary, _ = monte_carlo_weight_analysis(cbs_frame, n_samples=iterations, random_state=random_state)
    summary = summary.rename(columns={
        'mean_sampled_cbs': 'cbs_mean',
        'std_sampled_cbs': 'cbs_std',
        'top_model_frequency': 'top1_probability',
    })
    summary['top3_probability'] = np.nan
    summary['noise_std'] = np.nan
    return summary


def _render_significance_text(df: pd.DataFrame) -> str:
    if df.empty:
        return 'No results were produced.'
    top = df.sort_values('mean_absolute_delta_cbs', ascending=False).iloc[0]
    return (
        f"The most sensitive CBS component is {top['metric']} under {top['scenario']} perturbations, "
        f"with mean absolute CBS change {top['mean_absolute_delta_cbs']:.4f}."
    )


def _write_report(
    output_dir: Path,
    sensitivity: pd.DataFrame,
    sensitivity_ranking: pd.DataFrame,
    weight_robustness: pd.DataFrame,
    influence: pd.DataFrame,
    correlations: pd.DataFrame,
    redundancy: pd.DataFrame,
    normalization: pd.DataFrame,
    ranking_stability: pd.DataFrame,
) -> str:
    report = [
        '# CBS Validation Report',
        '',
        '## Summary',
        '- CBS was evaluated without changing the score definition.',
        '- Analysis covers perturbation sensitivity, weight robustness, metric influence, correlation structure, normalization behavior, and Monte Carlo ranking stability.',
        '',
        '## Sensitivity Analysis',
        _render_significance_text(sensitivity_ranking),
        '',
        '## Weight Robustness',
        f"Rank stability across perturbations: {weight_robustness['ranking_stability'].mean():.4f} average Spearman correlation." if not weight_robustness.empty else 'No weight robustness results.',
        '',
        '## Metric Dominance',
    ]

    if not influence.empty:
        best = influence.sort_values('mean_leave_one_out_impact', ascending=False).iloc[0]
        report.append(f"- Highest mean leave-one-out impact: {best['metric']} ({best['mean_leave_one_out_impact']:.4f}).")

    report.extend([
        '',
        '## Correlation Analysis',
        '- Pearson and Spearman correlations were computed between CBS and each component metric.',
    ])
    if not redundancy.empty:
        report.append(f"- Potentially redundant pairs (|Pearson r| >= 0.90): {len(redundancy)}")

    report.extend([
        '',
        '## Normalization Validation',
        '- Normalization was checked for score compression and rank distortions.',
        '',
        '## Monte Carlo Stability',
    ])
    if not ranking_stability.empty:
        report.append(
            f"- Mean CBS variance across models: {ranking_stability['cbs_std'].mean():.4f}; top-1 retention: {ranking_stability['top1_probability'].mean():.4f}."
        )

    report.extend([
        '',
        '## Acceptance Criteria',
        '- Robustness: assessed via perturbation and weight sensitivity.',
        '- Interpretability: assessed via contributions and dominance analysis.',
        '- Stability: assessed via Monte Carlo ranking consistency and confidence bounds.',
        '- Methodological soundness: assessed via correlations, normalization checks, and ranking distortion estimates.',
    ])

    path = output_dir / 'cbs_validation_report.md'
    path.write_text('\n'.join(report) + '\n', encoding='utf8')
    return str(path)


def _plot_tornado(sensitivity_ranking: pd.DataFrame, out_dir: Path) -> Tuple[str, str]:
    _require_matplotlib()
    fig, ax = plt.subplots(figsize=(9, 5))
    data = sensitivity_ranking[sensitivity_ranking['scenario'] == 'small_perturbation'].sort_values('mean_absolute_delta_cbs', ascending=True)
    ax.barh(data['metric'], data['mean_absolute_delta_cbs'], color='#3563e9')
    ax.set_title('CBS Sensitivity Tornado Plot')
    ax.set_xlabel('Mean absolute CBS change')
    fig.tight_layout()
    base = out_dir / 'tornado_plot'
    png = f'{base}.png'
    pdf = f'{base}.pdf'
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(pdf, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return png, pdf


def _plot_weight_sensitivity(weight_robustness: pd.DataFrame, out_dir: Path) -> Tuple[str, str]:
    _require_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 5))
    data = weight_robustness.copy()
    data['direction_sign'] = data['direction'].map({'increase': 1, 'decrease': -1})
    grouped = data.groupby(['metric', 'perturbation_level'], as_index=False)['ranking_stability'].mean()
    for metric, group in grouped.groupby('metric'):
        ax.plot(group['perturbation_level'] * 100, group['ranking_stability'], marker='o', label=metric)
    ax.set_title('CBS Weight Sensitivity')
    ax.set_xlabel('Weight perturbation (%)')
    ax.set_ylabel('Spearman rank correlation vs baseline')
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    base = out_dir / 'weight_sensitivity_plot'
    png = f'{base}.png'
    pdf = f'{base}.pdf'
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(pdf, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return png, pdf


def _plot_correlation_heatmap(cbs_frame: pd.DataFrame, out_dir: Path) -> Tuple[str, str]:
    _require_matplotlib()
    corr = cbs_frame[['cbs'] + CBS_METRICS].corr(method='pearson')
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap_name = 'vlag' if sns is not None else 'RdYlBu'
    if sns is not None:
        sns.heatmap(corr, annot=True, fmt='.2f', cmap=cmap_name, center=0.0, ax=ax)
    else:
        im = ax.imshow(corr.values, cmap=cmap_name, vmin=-1, vmax=1)
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha='right')
        ax.set_yticks(range(len(corr.index)))
        ax.set_yticklabels(corr.index)
    ax.set_title('CBS Correlation Heatmap')
    fig.tight_layout()
    base = out_dir / 'correlation_heatmap'
    png = f'{base}.png'
    pdf = f'{base}.pdf'
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(pdf, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return png, pdf


def _plot_stability_distributions(ranking_stability: pd.DataFrame, out_dir: Path) -> Tuple[str, str]:
    _require_matplotlib()
    fig, ax = plt.subplots(figsize=(9, 5))
    
    # Handle NaN values
    data_to_plot = ranking_stability['cbs_mean'].dropna()
    if len(data_to_plot) == 0:
        ax.text(0.5, 0.5, 'No valid stability data available', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
    else:
        ax.hist(data_to_plot, bins=min(10, len(data_to_plot)), alpha=0.8, color='#2f855a')
    
    ax.set_title('CBS Stability Distribution')
    ax.set_xlabel('Monte Carlo mean CBS')
    ax.set_ylabel('Count')
    fig.tight_layout()
    base = out_dir / 'stability_distribution'
    png = f'{base}.png'
    pdf = f'{base}.pdf'
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(pdf, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return png, pdf


def _plot_metric_contributions(influence: pd.DataFrame, out_dir: Path) -> Tuple[str, str]:
    _require_matplotlib()
    fig, ax = plt.subplots(figsize=(9, 5))
    
    # Handle empty or missing data
    if influence.empty or 'variance_contribution_pct' not in influence.columns:
        ax.text(0.5, 0.5, 'No metric contribution data available', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
    else:
        data = influence.sort_values('variance_contribution_pct', ascending=True)
        if len(data) > 0:
            ax.barh(data['metric'], data['variance_contribution_pct'] * 100.0, color='#805ad5')
    
    ax.set_title('CBS Metric Contribution To Variance')
    ax.set_xlabel('Contribution to CBS variance (%)')
    fig.tight_layout()
    base = out_dir / 'metric_contribution_plot'
    png = f'{base}.png'
    pdf = f'{base}.pdf'
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(pdf, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return png, pdf


def _plot_ranking_stability(ranking_stability: pd.DataFrame, out_dir: Path) -> Tuple[str, str]:
    _require_matplotlib()
    fig, ax = plt.subplots(figsize=(9, 5))
    
    # Handle empty or invalid data
    if ranking_stability.empty or 'mean_rank' not in ranking_stability.columns:
        ax.text(0.5, 0.5, 'No ranking stability data available', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
    else:
        # Filter out rows with NaN mean_rank
        data = ranking_stability.dropna(subset=['mean_rank']).sort_values('baseline_rank', na_position='last')
        if len(data) > 0:
            ax.errorbar(data.index, data['mean_rank'], 
                       yerr=data['rank_std'].fillna(0.0), 
                       fmt='o', color='#2b6cb0', capsize=4)
            ax.set_xticks(range(len(data)))
            ax.set_xticklabels(data.index, rotation=30)
            ax.invert_yaxis()
        else:
            ax.text(0.5, 0.5, 'No valid ranking data', 
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
    
    ax.set_title('Monte Carlo Weight Ranking Stability')
    ax.set_xlabel('Model')
    ax.set_ylabel('Mean rank under sampled weights')
    fig.tight_layout()
    base = out_dir / 'ranking_stability_plot'
    png = f'{base}.png'
    pdf = f'{base}.pdf'
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(pdf, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return png, pdf


def run_cbs_validation(results_dir: str | Path, output_dir: Optional[str | Path] = None, perturbation: float = 0.01, noise_std: float = 0.02, mc_iterations: int = 2000, random_state: int = 42) -> CBSValidationArtifacts:
    bundle = load_results_bundle(results_dir)
    normalized = bundle['normalized']
    raw = bundle['raw']

    out_root = Path(output_dir) if output_dir is not None else Path(results_dir) / 'cbs_validation'
    out_root.mkdir(parents=True, exist_ok=True)

    cbs_frame = build_cbs_frame(normalized)
    baseline_check = pd.merge(
        cbs_frame[['model', 'cbs']],
        bundle['cbs'][['model', 'cbs']].rename(columns={'cbs': 'saved_cbs'}),
        on='model',
        how='left',
    )
    cbs_frame['saved_cbs'] = baseline_check['saved_cbs']
    cbs_frame['cbs_delta_vs_saved'] = cbs_frame['cbs'] - cbs_frame['saved_cbs']

    fold_std = None
    if not raw.empty and 'stability__assessment__primary_metric_cv' in raw.columns:
        fold_std = raw.set_index('model')['stability__assessment__primary_metric_cv'].astype(float)

    sensitivity, sensitivity_ranking = sensitivity_analysis(cbs_frame, perturbation=perturbation, noise_std=noise_std, fold_std=fold_std, random_state=random_state)
    weight_robustness = weight_robustness_analysis(cbs_frame, weights=CBS_WEIGHTS)
    influence = metric_dominance_analysis(cbs_frame)
    correlations, redundancy = correlation_analysis(cbs_frame)
    normalization = normalization_validation(raw, cbs_frame)
    ranking_stability, monte_carlo_samples = monte_carlo_weight_analysis(cbs_frame, n_samples=mc_iterations, random_state=random_state)
    ranking_stability = ranking_stability.rename(columns={
        'mean_sampled_cbs': 'cbs_mean',
        'std_sampled_cbs': 'cbs_std',
        'top_model_frequency': 'top1_probability',
    })

    sensitivity_path = out_root / 'sensitivity_analysis.csv'
    weight_path = out_root / 'weight_robustness.csv'
    influence_path = out_root / 'metric_influence.csv'
    ranking_path = out_root / 'ranking_stability.csv'
    mc_samples_path = out_root / 'monte_carlo_weight_samples.csv'
    correlation_path = out_root / 'correlation_analysis.csv'
    normalization_path = out_root / 'normalization_validation.csv'

    sensitivity.to_csv(sensitivity_path, index=False)
    weight_robustness.to_csv(weight_path, index=False)
    influence.to_csv(influence_path, index=False)
    ranking_stability.to_csv(ranking_path, index=False)
    monte_carlo_samples.to_csv(mc_samples_path, index=False)
    correlations.to_csv(correlation_path, index=False)
    normalization.to_csv(normalization_path, index=False)
    if not redundancy.empty:
        redundancy.to_csv(out_root / 'redundancy_pairs.csv', index=False)

    plots_dir = out_root / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    plots = {
        'sensitivity_plot': _plot_tornado(sensitivity_ranking, plots_dir),
        'weight_impact_plot': _plot_weight_sensitivity(weight_robustness, plots_dir),
        'ranking_stability_plot': _plot_ranking_stability(ranking_stability, plots_dir),
        'metric_contribution_plot': _plot_metric_contributions(influence, plots_dir),
        'correlation_heatmap': _plot_correlation_heatmap(cbs_frame, plots_dir),
        'stability_distribution': _plot_stability_distributions(ranking_stability, plots_dir),
    }

    plot_paths = {name: pair[0] for name, pair in plots.items()}
    report_path = write_cbs_validation_report(
        out_root / 'cbs_validation_report.md',
        dataset_name=str(results_dir),
        baseline=cbs_frame,
        weight_robustness=weight_robustness,
        dominance=influence,
        normalization=normalization,
        monte_carlo=ranking_stability,
        plots=plot_paths,
    )

    return CBSValidationArtifacts(
        report_path=report_path,
        sensitivity_path=str(sensitivity_path),
        weight_robustness_path=str(weight_path),
        metric_influence_path=str(influence_path),
        ranking_stability_path=str(ranking_path),
        monte_carlo_samples_path=str(mc_samples_path),
        correlation_path=str(correlation_path),
        normalization_path=str(normalization_path),
        plots=plots,
    )
