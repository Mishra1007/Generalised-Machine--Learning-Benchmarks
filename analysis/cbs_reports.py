"""CBS validation report rendering helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd


def _fmt(value: object, digits: int = 4) -> str:
    try:
        number = float(value)
    except Exception:
        return str(value)
    if pd.isna(number):
        return 'nan'
    return f'{number:.{digits}f}'


def render_cbs_validation_report(
    *,
    dataset_name: str,
    baseline: pd.DataFrame,
    weight_robustness: pd.DataFrame,
    dominance: pd.DataFrame,
    normalization: pd.DataFrame,
    monte_carlo: pd.DataFrame,
    plots: Optional[Dict[str, str]] = None,
) -> str:
    lines = [
        '# CBS Validation Report',
        '',
        f'Dataset artifact: `{dataset_name}`',
        '',
        '## Methodology',
        '',
        'CBS was analyzed without changing the canonical weight vector or benchmark execution logic. The validation recomputes CBS from normalized component metrics, perturbs weights at +/-5%, +/-10%, and +/-20%, estimates rank robustness with Spearman and Kendall correlations, measures metric contribution/dominance, audits normalization behavior, and samples valid non-negative random weight vectors that sum to 1.',
        '',
        '## Baseline Ranking',
        '',
    ]

    for _, row in baseline.sort_values(['baseline_rank', 'model']).iterrows():
        lines.append(f"- Rank {_fmt(row['baseline_rank'], 1)}: {row['model']} CBS={_fmt(row['cbs'])}")

    lines.extend(['', '## Sensitivity And Robustness', ''])
    if weight_robustness.empty:
        lines.append('No weight perturbation results were produced.')
    else:
        grouped = (
            weight_robustness.groupby('perturbation_level', as_index=False)
            .agg(
                mean_spearman=('spearman_rank_correlation', 'mean'),
                mean_kendall=('kendall_rank_correlation', 'mean'),
                mean_reversals=('ranking_reversals', 'mean'),
                max_abs_rank_shift=('rank_shift', lambda s: float(s.abs().max())),
            )
            .sort_values('perturbation_level')
        )
        lines.append('| Perturbation | Mean Spearman | Mean Kendall | Mean Pairwise Reversals | Max Abs Rank Shift |')
        lines.append('|---:|---:|---:|---:|---:|')
        for _, row in grouped.iterrows():
            lines.append(
                f"| {100 * row['perturbation_level']:.0f}% | {_fmt(row['mean_spearman'])} | {_fmt(row['mean_kendall'])} | {_fmt(row['mean_reversals'])} | {_fmt(row['max_abs_rank_shift'])} |"
            )

    lines.extend(['', '## Metric Dominance', ''])
    if dominance.empty:
        lines.append('No dominance results were produced.')
    else:
        top = dominance.sort_values('variance_contribution_pct', ascending=False).iloc[0]
        lines.append(
            f"Highest CBS variance contribution: `{top['metric']}` ({_fmt(100 * top['variance_contribution_pct'], 2)}%)."
        )
        lines.append('')
        lines.append('| Metric | Weight | Mean Contribution | Variance Contribution % | Spearman With CBS | Leave-One-Out Impact |')
        lines.append('|---|---:|---:|---:|---:|---:|')
        for _, row in dominance.sort_values('variance_contribution_pct', ascending=False).iterrows():
            lines.append(
                f"| {row['metric']} | {_fmt(row['weight'])} | {_fmt(row['mean_contribution'])} | {_fmt(100 * row['variance_contribution_pct'], 2)} | {_fmt(row['spearman_cbs_corr'])} | {_fmt(row['mean_leave_one_out_impact'])} |"
            )

    lines.extend(['', '## Normalization Audit', ''])
    if normalization.empty:
        lines.append('No raw results were available for normalization auditing.')
    else:
        artifacts = normalization[normalization['artifact_flag']]
        if artifacts.empty:
            lines.append('No boundedness failures, constant-metric compression flags, or rank distortions were detected in the audited artifact.')
        else:
            lines.append('Normalization artifacts flagged:')
            for _, row in artifacts.iterrows():
                lines.append(f"- {row['metric']}: {row['artifact_reason']}")

    lines.extend(['', '## Monte Carlo Weight Analysis', ''])
    if monte_carlo.empty:
        lines.append('No Monte Carlo weight results were produced.')
    else:
        top_frequency_col = 'top_model_frequency' if 'top_model_frequency' in monte_carlo.columns else 'top1_probability'
        top_model = monte_carlo.sort_values(top_frequency_col, ascending=False).iloc[0]
        lines.append(
            f"Most frequent top model under sampled valid weights: `{top_model['model']}` ({_fmt(100 * top_model[top_frequency_col], 2)}%)."
        )
        lines.append('')
        lines.append('| Model | Baseline Rank | Mean Rank | Rank Std | Top Model Frequency | Rank Reversal Frequency |')
        lines.append('|---|---:|---:|---:|---:|---:|')
        for _, row in monte_carlo.sort_values('baseline_rank').iterrows():
            lines.append(
                f"| {row['model']} | {_fmt(row['baseline_rank'], 1)} | {_fmt(row['mean_rank'])} | {_fmt(row['rank_std'])} | {_fmt(100 * row[top_frequency_col], 2)}% | {_fmt(100 * row['rank_reversal_frequency'], 2)}% |"
            )

    lines.extend(['', '## Recommendations', ''])
    lines.append('- Preserve CBS weights unless a domain-specific justification is documented; this report is diagnostic only.')
    lines.append('- Treat min-max normalization as dataset-relative; compare CBS across datasets only with caution.')
    lines.append('- Report robustness artifacts alongside CBS rankings in publication material.')
    lines.append('- Investigate any future dataset where one component contributes most CBS variance or where Monte Carlo top-model frequency is diffuse.')

    if plots:
        lines.extend(['', '## Visualizations', ''])
        for name, path in plots.items():
            lines.append(f"- {name}: `{path}`")

    return '\n'.join(lines) + '\n'


def write_cbs_validation_report(path: str | Path, **kwargs) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_cbs_validation_report(**kwargs), encoding='utf8')
    return str(target)
