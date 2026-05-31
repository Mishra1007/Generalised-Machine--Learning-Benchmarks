"""Pre-import launcher for reproducible benchmark runs.

Run this instead of importing framework modules directly when you need the
hash seed and thread controls applied before NumPy / scikit-learn import.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from runtime_bootstrap import apply_early_binding


def _load_seed_from_config(config_path: str | None) -> int | None:
    if not config_path:
        return None

    path = Path(config_path)
    if not path.exists():
        return None

    with path.open('r', encoding='utf8') as fh:
        cfg = yaml.safe_load(fh) or {}

    seed = cfg.get('random_state', cfg.get('seed'))
    return seed if isinstance(seed, int) else None


def main() -> None:
    parser = argparse.ArgumentParser(description='Reproducible launcher for the benchmark framework')
    parser.add_argument('--config', '-c', help='Path to experiment YAML config')
    parser.add_argument('--out', '-o', default='results', help='Output directory for results')
    args = parser.parse_args()

    seed = _load_seed_from_config(args.config)
    apply_early_binding(seed=seed)

    from main import main as framework_main

    import sys

    argv = ['main.py']
    if args.config:
        argv.extend(['--config', args.config])
    argv.extend(['--out', args.out])
    sys.argv = argv
    framework_main()


if __name__ == '__main__':
    main()
