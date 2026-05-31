"""Result storage and reproducible logging for benchmark runs.

Creates a results/<dataset>/ folder with:
- raw_results.csv
- normalized_results.csv
- cbs_scores.csv
- plots/ (saved plots)
- logs.txt (structured JSON lines)
- config.json (experiment config)

Design notes:
- Records environment info (python, platform, git commit if available).
- Uses structured JSON-lines logging for traceability.
"""

from typing import Dict, Any, Tuple
import os
import json
import csv
import sys
import datetime
import logging
import subprocess
import platform
from pathlib import Path

try:
    # Python 3.8+
    import importlib.metadata as importlib_metadata
except Exception:
    import importlib_metadata  # type: ignore

from metrics.normalization import normalize_model_summaries
from metrics.cbs import compute_cbs
from metrics.visualization import generate_plots_for_dataset

LOG_FILENAME = 'logs.txt'


def _git_info(project_root: str) -> Dict[str, Any]:
    info = {}
    try:
        # Get current commit hash
        commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=project_root, stderr=subprocess.DEVNULL)
        info['git_commit'] = commit.decode().strip()
        # Check for uncommitted changes
        status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=project_root)
        info['git_dirty'] = bool(status.strip())
    except Exception:
        info['git_commit'] = None
        info['git_dirty'] = None
    return info


def _env_info() -> Dict[str, Any]:
    pkgs = []
    try:
        for dist in importlib_metadata.distributions():
            pkgs.append({'name': dist.metadata['Name'] if 'Name' in dist.metadata else dist.name, 'version': dist.version})
    except Exception:
        pkgs = []

    return {
        'python_version': sys.version.replace('\n', ' '),
        'platform': platform.platform(),
        'installed_packages': pkgs,
    }


class JSONLineLogger:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self._fh = open(filepath, 'a', encoding='utf8')

    def log(self, level: str, message: str, **meta):
        entry = {
            'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
            'level': level,
            'message': message,
            'meta': meta,
        }
        self._fh.write(json.dumps(entry, default=str) + '\n')
        self._fh.flush()

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass


def _flatten_summary(models_summaries: Dict[str, Dict[str, Any]]) -> Tuple[list, list]:
    """Flatten model summaries into CSV-friendly rows and columns.

    Returns (columns, rows) where rows are dicts mapping column->value.
    """
    rows = []
    columns = set()
    for model_name, summary in models_summaries.items():
        row = {'model': model_name}
        # overall metrics
        overall = summary.get('overall', {})
        for k, v in overall.items():
            key = f'overall__{k}'
            row[key] = v
            columns.add(key)
        # stability
        stability = summary.get('stability', {})
        for k, v in stability.items():
            if isinstance(v, dict):
                for subk, subv in v.items():
                    key = f'stability__{k}__{subk}'
                    row[key] = subv
                    columns.add(key)
            else:
                key = f'stability__{k}'
                row[key] = v
                columns.add(key)
        rows.append(row)
    columns = ['model'] + sorted(list(columns))
    return columns, rows


def save_experiment_results(dataset_name: str, models_summaries: Dict[str, Dict[str, Any]], out_root: str = 'results', config: Dict[str, Any] = None, seed: int = None) -> Dict[str, str]:
    """Save raw results, normalized metrics, CBS, plots, config, and logs.

    Returns mapping of saved file paths.
    """
    project_root = os.getcwd()
    base = os.path.join(out_root, dataset_name)
    plots_dir = os.path.join(base, 'plots')
    os.makedirs(base, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    logger = JSONLineLogger(os.path.join(base, LOG_FILENAME))
    logger.log('INFO', 'Starting experiment save', dataset=dataset_name)

    # Env & VCS
    git = _git_info(project_root)
    env = _env_info()
    logger.log('INFO', 'Environment captured', git=git, python=env.get('python_version'))

    # Save config
    cfg = config or {}
    meta = {
        'dataset': dataset_name,
        'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
        'seed': seed,
        'project_root': project_root,
        'git': git,
        'env': {'python_version': env.get('python_version'), 'platform': env.get('platform')},
    }
    cfg_with_meta = {'config': cfg, 'meta': meta}
    cfg_path = os.path.join(base, 'config.json')
    with open(cfg_path, 'w', encoding='utf8') as fh:
        json.dump(cfg_with_meta, fh, indent=2, default=str)
    logger.log('INFO', 'Saved experiment config', path=cfg_path)

    # Raw results
    raw_path = os.path.join(base, 'raw_results.csv')
    cols, rows = _flatten_summary(models_summaries)
    with open(raw_path, 'w', newline='', encoding='utf8') as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    logger.log('INFO', 'Saved raw results', path=raw_path, models=len(rows))

    # Normalized
    normalized = normalize_model_summaries(models_summaries)
    norm_path = os.path.join(base, 'normalized_results.csv')
    # normalized is dict model->{metric: val}
    norm_cols = ['model'] + sorted(next(iter(normalized.values())).keys() if normalized else [])
    with open(norm_path, 'w', newline='', encoding='utf8') as fh:
        writer = csv.DictWriter(fh, fieldnames=norm_cols)
        writer.writeheader()
        for m, metrics in normalized.items():
            row = {'model': m}
            row.update(metrics)
            writer.writerow(row)
    logger.log('INFO', 'Saved normalized results', path=norm_path)

    # CBS
    cbs_map = compute_cbs(models_summaries)
    cbs_path = os.path.join(base, 'cbs_scores.csv')
    # Each entry: model, cbs, plus normalized metrics
    cbs_cols = ['model', 'cbs'] + sorted(next(iter(normalized.values())).keys() if normalized else [])
    with open(cbs_path, 'w', newline='', encoding='utf8') as fh:
        writer = csv.DictWriter(fh, fieldnames=cbs_cols)
        writer.writeheader()
        for m, entry in cbs_map.items():
            row = {'model': m, 'cbs': entry.get('cbs')}
            row.update(entry.get('normalized', {}))
            writer.writerow(row)
    logger.log('INFO', 'Saved CBS scores', path=cbs_path)

    # Plots: use visualization but point to results/<dataset>/plots
    plots = generate_plots_for_dataset(dataset_name, models_summaries, out_dir=os.path.join(out_root))
    # The visualization saves into out_dir/<dataset>/, so move or confirm path
    # We expect plots under out_root/dataset_name/
    logger.log('INFO', 'Generated plots', plots_dir=os.path.join(out_root, dataset_name, 'plots'))

    logger.log('INFO', 'Experiment save complete', saved_files={'raw': raw_path, 'normalized': norm_path, 'cbs': cbs_path, 'config': cfg_path})
    logger.close()

    return {'raw': raw_path, 'normalized': norm_path, 'cbs': cbs_path, 'config': cfg_path, 'logs': os.path.join(base, LOG_FILENAME), 'plots_dir': os.path.join(out_root, dataset_name)}


if __name__ == '__main__':
    print('Run save_experiment_results() from your code')
