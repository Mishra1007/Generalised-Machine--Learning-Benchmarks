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
import tempfile
from pathlib import Path

try:
    # Python 3.8+
    import importlib.metadata as importlib_metadata
except Exception:
    import importlib_metadata  # type: ignore

import pandas as pd

from metrics.normalization import normalize_model_summaries
from metrics.cbs import compute_cbs
from metrics.visualization import generate_plots_for_dataset
from reproducibility import build_experiment_manifest, render_reproducibility_report, validate_manifest
from analysis import (
    bootstrap_confidence_interval,
    effect_size_summary,
    global_significance_analysis,
    mean_confidence_interval,
    write_significance_artifacts,
)

LOG_FILENAME = 'logs.txt'


class MetadataValidationError(ValueError):
    pass


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


def validate_metadata_schema(metadata: Dict[str, Any]) -> None:
    required_fields = {
        'experiment_id': str,
        'run_id': str,
        'timestamp': str,
        'framework_version': str,
        'git_commit': (str, type(None)),
        'dataset_name': str,
        'dataset_source': (str, type(None)),
        'dataset_hash': str,
        'dataset_size': int,
        'feature_count': int,
        'target_column': (str, type(None)),
        'preprocessing_pipeline': dict,
        'model_names': list,
        'model_hyperparameters': dict,
        'validation_strategy': dict,
        'folds': list,
        'repetitions': int,
        'random_seed': int,
        'config_snapshot': dict,
        'python_version': str,
        'library_versions': dict,
        'class_distribution': dict,
        'environment': dict,
    }

    missing = [key for key in required_fields if key not in metadata]
    if missing:
        raise MetadataValidationError(f"Missing required metadata fields: {missing}")

    def _type_name(expected: Any) -> str:
        if isinstance(expected, tuple):
            return ' or '.join(t.__name__ for t in expected)
        return expected.__name__

    for field_name, expected_type in required_fields.items():
        value = metadata.get(field_name)
        if field_name in {'framework_version', 'python_version'}:
            if not isinstance(value, str) or not value.strip():
                raise MetadataValidationError(f"Field '{field_name}' must be a non-empty string")
            continue
        if not isinstance(value, expected_type):
            raise MetadataValidationError(f"Field '{field_name}' must be of type {_type_name(expected_type)}")

    environment = metadata['environment']
    required_environment = {
        'python_version': str,
        'python_executable': str,
        'python_implementation': str,
        'os': dict,
        'cpu': dict,
        'pip_freeze': list,
        'library_versions': dict,
        'packages': dict,
        'python_hash_seed': (str, type(None)),
        'thread_environment': dict,
        'launcher_environment': dict,
    }
    missing_environment = [key for key in required_environment if key not in environment]
    if missing_environment:
        raise MetadataValidationError(f"Missing required environment fields: {missing_environment}")

    package_names = ['numpy', 'pandas', 'scikit-learn', 'scipy', 'joblib']
    packages = environment.get('packages', {})
    for package_name in package_names:
        if package_name not in packages:
            raise MetadataValidationError(f"Missing required package version for '{package_name}'")

    thread_environment = environment.get('thread_environment', {})
    for thread_name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
        if thread_name not in thread_environment:
            raise MetadataValidationError(f"Missing required thread environment variable '{thread_name}'")


def _atomic_json_dump(path: str, payload: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=target.name + '.', suffix='.tmp', dir=str(target.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf8') as fh:
            json.dump(payload, fh, indent=2, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except Exception:
            pass
        raise


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


def _build_score_matrix(validation_results: Dict[str, Any], primary_metric: str = 'accuracy'):
    if not validation_results:
        return None, []

    model_names = []
    score_maps = []
    for model_name, vres in validation_results.items():
        if vres is None:
            continue
        fold_results = getattr(vres, 'fold_results', None) or []
        scores = {}
        for fold in fold_results:
            metrics = getattr(fold, 'metrics', {}) or {}
            if primary_metric in metrics:
                fold_key = (getattr(fold, 'repetition_id', None), getattr(fold, 'fold_id', None))
                if fold_key in scores:
                    raise ValueError(f'Duplicate fold observation key for model {model_name}: {fold_key}')
                scores[fold_key] = metrics[primary_metric]
        if scores:
            model_names.append(model_name)
            score_maps.append(scores)

    if len(model_names) < 2:
        return None, model_names

    common_keys = set(score_maps[0].keys())
    for score_map in score_maps[1:]:
        common_keys &= set(score_map.keys())

    if not common_keys:
        return None, model_names

    ordered_keys = sorted(common_keys)
    # Matrix shape: (n_observations, n_models)
    matrix = [[score_map[key] for score_map in score_maps] for key in ordered_keys]
    if matrix and len(matrix[0]) != len(model_names):
        raise ValueError('Score matrix column count must match number of models')
    return matrix, model_names


def save_experiment_results(dataset_name: str, models_summaries: Dict[str, Dict[str, Any]], out_root: str = 'results', config: Dict[str, Any] = None, seed: int = None, metadata: Dict[str, Any] = None, validation_results: Dict[str, Any] = None) -> Dict[str, str]:
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
    _atomic_json_dump(cfg_path, cfg_with_meta)
    logger.log('INFO', 'Saved experiment config', path=cfg_path)

    # Save metadata.json (audit-friendly schema)
    if metadata is not None:
        try:
            validate_metadata_schema(metadata)

            meta_path = os.path.join(base, 'metadata.json')
            _atomic_json_dump(meta_path, metadata)
            logger.log('INFO', 'Saved experiment metadata', path=meta_path)
        except Exception as e:
            logger.log('ERROR', 'Failed to save metadata.json', error=str(e))
            # Fail fast to enforce reproducibility guarantees
            raise

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

    # Save metrics.json (structured metrics per model)
    try:
        metrics_path = os.path.join(base, 'metrics.json')
        _atomic_json_dump(metrics_path, models_summaries)
        logger.log('INFO', 'Saved metrics summary JSON', path=metrics_path)
    except Exception:
        logger.log('WARN', 'Failed to save metrics.json')

    # If validation_results provided and contains predictions, write predictions per model
    if validation_results:
        try:
            # validation_results is mapping model_name -> ValidationResults object
            for model_name, vres in validation_results.items():
                if vres is None:
                    continue
                # If it's a ValidationResults instance, extract fold_results
                try:
                    fold_results = getattr(vres, 'fold_results', None)
                except Exception:
                    fold_results = None

                if not fold_results:
                    continue

                # Write predictions CSV: repetition_id, fold_id, test_index, y_test, y_pred
                pred_path = os.path.join(base, f'predictions_{model_name}.csv')
                with open(pred_path, 'w', newline='', encoding='utf8') as fh:
                    writer = csv.writer(fh)
                    writer.writerow(['repetition_id', 'fold_id', 'test_index', 'y_test', 'y_pred'])
                    for fr in fold_results:
                        test_indices = fr.test_indices or []
                        y_test = fr.y_test.tolist() if fr.y_test is not None else []
                        y_pred = fr.y_pred.tolist() if fr.y_pred is not None else []
                        for idx, yt, yp in zip(test_indices, y_test, y_pred):
                            writer.writerow([fr.repetition_id, fr.fold_id, idx, yt, yp])
                logger.log('INFO', 'Saved predictions for model', model=model_name, path=pred_path)
        except Exception:
            logger.log('WARN', 'Failed to save predictions')

    # Plots: use visualization but point to results/<dataset>/plots
    plots = generate_plots_for_dataset(dataset_name, models_summaries, out_dir=os.path.join(out_root))
    # The visualization saves into out_dir/<dataset>/, so move or confirm path
    # We expect plots under out_root/dataset_name/
    logger.log('INFO', 'Generated plots', plots_dir=os.path.join(out_root, dataset_name, 'plots'))

    manifest_path = None
    report_path = None
    significance_paths = {}
    if metadata is not None:
        manifest = build_experiment_manifest(
            metadata=metadata,
            config=cfg,
            artifacts={
                'config': cfg_path,
                'metadata': os.path.join(base, 'metadata.json') if metadata is not None else None,
                'raw_results': raw_path,
                'normalized_results': norm_path,
                'cbs_scores': cbs_path,
                'metrics_json': os.path.join(base, 'metrics.json'),
                'plots_dir': os.path.join(out_root, dataset_name, 'plots'),
            },
            project_root=project_root,
        )
        validate_manifest(manifest)
        manifest_path = os.path.join(base, 'experiment_manifest.json')
        _atomic_json_dump(manifest_path, manifest)
        report_path = os.path.join(base, 'reproducibility_report.md')
        with open(report_path, 'w', encoding='utf8') as fh:
            fh.write(render_reproducibility_report(manifest))
        logger.log('INFO', 'Saved experiment manifest and report', manifest=manifest_path, report=report_path)

    if validation_results:
        try:
            matrix, model_names = _build_score_matrix(validation_results, primary_metric='accuracy')
            if matrix is not None and len(model_names) >= 2:
                import numpy as np

                analysis = global_significance_analysis(np.asarray(matrix, dtype=float), model_names=model_names)

                ci_rows = []
                for idx, model_name in enumerate(model_names):
                    values = np.asarray(matrix, dtype=float)[:, idx]
                    mean_ci = mean_confidence_interval(values, confidence=0.95)
                    boot_ci = bootstrap_confidence_interval(values, confidence=0.95)
                    ci_rows.append({
                        'model': model_name,
                        'mean': mean_ci['mean'],
                        'mean_ci_lower': mean_ci['lower'],
                        'mean_ci_upper': mean_ci['upper'],
                        'bootstrap_ci_lower': boot_ci['lower'],
                        'bootstrap_ci_upper': boot_ci['upper'],
                    })

                pairwise = analysis['pairwise_wilcoxon'].copy()
                effect_rows = []
                for _, row in pairwise.iterrows():
                    idx_a = model_names.index(row['model_a'])
                    idx_b = model_names.index(row['model_b'])
                    matrix_arr = np.asarray(matrix, dtype=float)
                    effect = effect_size_summary(matrix_arr[:, idx_a], matrix_arr[:, idx_b], paired=True)
                    effect_rows.append(effect)
                if effect_rows:
                    effect_df = pd.DataFrame(effect_rows)
                    pairwise = pd.concat([pairwise.reset_index(drop=True), effect_df.reset_index(drop=True)], axis=1)
                    analysis['pairwise_wilcoxon'] = pairwise

                if ci_rows:
                    analysis['confidence_intervals'] = pd.DataFrame(ci_rows)

                significance_paths = write_significance_artifacts(analysis, base)
                logger.log('INFO', 'Saved statistical significance artifacts', **significance_paths)
        except Exception as exc:
            logger.log('WARN', 'Failed to generate statistical significance artifacts', error=str(exc))

    saved_files = {'raw': raw_path, 'normalized': norm_path, 'cbs': cbs_path, 'config': cfg_path, 'manifest': manifest_path, 'report': report_path, **significance_paths}
    logger.log('INFO', 'Experiment save complete', saved_files=saved_files)
    logger.close()

    return {'raw': raw_path, 'normalized': norm_path, 'cbs': cbs_path, 'config': cfg_path, 'manifest': manifest_path, 'report': report_path, **significance_paths, 'logs': os.path.join(base, LOG_FILENAME), 'plots_dir': os.path.join(out_root, dataset_name)}


if __name__ == '__main__':
    print('Run save_experiment_results() from your code')
