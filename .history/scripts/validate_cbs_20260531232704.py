#!/usr/bin/env python
"""Generate CBS validation artifacts from a saved benchmark results folder."""

from __future__ import annotations

import argparse
from pathlib import Path

from analysis.cbs_validation import run_cbs_validation


def main() -> int:
    parser = argparse.ArgumentParser(description='Validate CBS robustness and stability from saved benchmark outputs.')
    parser.add_argument('results_dir', help='Path to a benchmark results directory containing normalized_results.csv and cbs_scores.csv')
    parser.add_argument('--output-dir', default=None, help='Directory for CBS validation outputs (defaults to <results_dir>/cbs_validation)')
    parser.add_argument('--perturbation', type=float, default=0.01, help='Absolute metric perturbation for sensitivity analysis')
    parser.add_argument('--noise-std', type=float, default=0.02, help='Noise standard deviation used in robustness simulations')
    parser.add_argument('--mc-iterations', type=int, default=2000, help='Monte Carlo iterations for ranking stability')
    args = parser.parse_args()

    artifacts = run_cbs_validation(
        results_dir=Path(args.results_dir),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        perturbation=args.perturbation,
        noise_std=args.noise_std,
        mc_iterations=args.mc_iterations,
    )

    print('CBS validation complete')
    print(f'Report: {artifacts.report_path}')
    print(f'Sensitivity: {artifacts.sensitivity_path}')
    print(f'Weight robustness: {artifacts.weight_robustness_path}')
    print(f'Metric influence: {artifacts.metric_influence_path}')
    print(f'Ranking stability: {artifacts.ranking_stability_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
