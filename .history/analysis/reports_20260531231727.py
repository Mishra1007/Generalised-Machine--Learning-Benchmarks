"""Reporting helpers for statistical significance analyses."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from analysis.effect_size import effect_size_summary


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f'{value:.6g}'
    return str(value)


def build_significance_report(analysis: Dict[str, Any]) -> str:
    friedman = analysis.get('friedman', {})
    nemenyi = analysis.get('nemenyi', {})
    pairwise = analysis.get('pairwise_wilcoxon', pd.DataFrame())
    rankings = analysis.get('ranking_table', pd.DataFrame())

    lines = [
        '# Significance Report',
        '',
        '## Test Assumptions',
        '- Friedman: paired comparisons across the same datasets/folds.',
        '- Wilcoxon Signed-Rank: paired sample differences per model pair.',
        '- Nemenyi: post-hoc multiple comparison after Friedman.',
        '',
        '## Statistical Outputs',
        f"Friedman statistic: {_format_value(friedman.get('statistic'))}",
        f"Friedman p-value: {_format_value(friedman.get('p_value'))}",
        f"Friedman interpretation: {friedman.get('interpretation')}",
        f"Nemenyi critical difference: {_format_value(nemenyi.get('critical_difference'))}",
        '',
        '## Conclusions',
        'Use the comparison_table.csv for pairwise significance and ranking_table.csv for publication-ready ordering.',
    ]

    if isinstance(pairwise, pd.DataFrame) and not pairwise.empty:
        lines.extend(['', '## Pairwise Wilcoxon Summary'])
        for _, row in pairwise.iterrows():
            lines.append(
                f"- {row['model_a']} vs {row['model_b']}: p={_format_value(row['p_value'])}, significant={row['significant']}"
            )

    if isinstance(rankings, pd.DataFrame) and not rankings.empty:
        lines.extend(['', '## Ranking Table'])
        for _, row in rankings.iterrows():
            lines.append(f"- {row['model']}: rank={_format_value(row.iloc[1]) if len(row) > 1 else _format_value(row['rank'])}")

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
    pairwise = analysis.get('pairwise_wilcoxon', pd.DataFrame())
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
