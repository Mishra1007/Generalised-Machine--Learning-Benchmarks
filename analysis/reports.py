"""Reporting helpers for statistical significance analyses."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f'{value:.6g}'
    return str(value)


def build_significance_report(analysis: Dict[str, Any]) -> str:
    significance_method = analysis.get('significance_method', '')
    is_single_dataset = significance_method == 'corrected_resampled_t_test' or 'pairwise_corrected_t' in analysis

    friedman = analysis.get('friedman', {})
    nemenyi = analysis.get('nemenyi', {})
    pairwise = analysis.get('pairwise_corrected_t', analysis.get('pairwise_wilcoxon', pd.DataFrame()))
    rankings = analysis.get('ranking_table', pd.DataFrame())
    intervals = analysis.get('confidence_intervals', pd.DataFrame())

    lines = [
        '# Significance Report',
        '',
        '## Test Assumptions',
    ]

    if is_single_dataset:
        lines.extend([
            '- Corrected Resampled t-test (Nadeau-Bengio): protects against inflated Type I error due to overlapping training sets in repeated k-fold cross-validation.',
            '- Holm-Bonferroni: controls Family-Wise Error Rate (FWER) across multiple pairwise comparisons.',
            '- Note: Friedman and Nemenyi tests are strictly omitted for single-dataset benchmarks because cross-validation folds violate the independence assumptions required by these tests.',
        ])
    else:
        lines.extend([
            '- Friedman: paired comparisons across independent datasets.',
            '- Wilcoxon Signed-Rank: paired sample differences per model pair across independent datasets.',
            '- Nemenyi: post-hoc multiple comparison after Friedman.',
            '',
            '## Statistical Outputs',
            f"Friedman statistic: {_format_value(friedman.get('statistic'))}",
            f"Friedman p-value: {_format_value(friedman.get('p_value'))}",
            f"Friedman interpretation: {friedman.get('interpretation')}",
            f"Nemenyi critical difference: {_format_value(nemenyi.get('critical_difference'))}",
        ])

    lines.extend([
        '',
        '## Conclusions',
        'Use the comparison_table.csv for pairwise significance and ranking_table.csv for publication-ready ordering.',
    ])

    if isinstance(intervals, pd.DataFrame) and not intervals.empty:
        lines.extend(['', '## Confidence Intervals'])
        for _, row in intervals.iterrows():
            lines.append(
                f"- {row['model']}: mean={_format_value(row['mean'])}, 95% CI=[{_format_value(row['mean_ci_lower'])}, {_format_value(row['mean_ci_upper'])}]"
            )

    if isinstance(pairwise, pd.DataFrame) and not pairwise.empty:
        summary_title = 'Pairwise Corrected t-test Summary' if is_single_dataset else 'Pairwise Wilcoxon Summary'
        lines.extend(['', f'## {summary_title}'])
        for _, row in pairwise.iterrows():
            raw_p = row.get('p_value_raw', row.get('p_value'))
            adj_p = row.get('p_value_adj', None)
            if adj_p is not None and adj_p == adj_p:
                lines.append(
                    f"- {row['model_a']} vs {row['model_b']}: raw p={_format_value(raw_p)}, adj p={_format_value(adj_p)}, significant={row['significant']}"
                )
            else:
                lines.append(
                    f"- {row['model_a']} vs {row['model_b']}: p={_format_value(raw_p)}, significant={row['significant']}"
                )

    if isinstance(rankings, pd.DataFrame) and not rankings.empty:
        lines.extend(['', '## Ranking Table'])
        for _, row in rankings.iterrows():
            rank_value = row['rank'] if 'rank' in row else row.get('average_rank')
            lines.append(f"- {row['model']}: rank={_format_value(rank_value)}")

    return '\n'.join(lines) + '\n'


def write_significance_artifacts(
    analysis: Dict[str, Any],
    out_dir: str | Path,
) -> Dict[str, str]:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)

    report_path = target / 'significance_report.md'
    comparison_path = target / 'comparison_table.csv'
    ranking_path = target / 'ranking_table.csv'

    report_path.write_text(build_significance_report(analysis), encoding='utf8')
    pairwise = analysis.get('pairwise_corrected_t', analysis.get('pairwise_wilcoxon', pd.DataFrame()))
    rankings = analysis.get('ranking_table', pd.DataFrame())
    if isinstance(pairwise, pd.DataFrame):
        pairwise.to_csv(comparison_path, index=False)
    else:
        pd.DataFrame(pairwise).to_csv(comparison_path, index=False)
    if isinstance(rankings, pd.DataFrame):
        rankings.to_csv(ranking_path, index=False)
    else:
        pd.DataFrame(rankings).to_csv(ranking_path, index=False)

    return {
        'report': str(report_path),
        'comparison_table': str(comparison_path),
        'ranking_table': str(ranking_path),
    }
