"""Build reproducibility manifests from benchmark execution metadata."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

from experiments.metadata import compute_dataset_fingerprint
from reproducibility.environment import capture_environment, capture_git_info


def _extract_model_section(metadata: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    model_names = metadata.get('model_names', [])
    hyperparameters = metadata.get('model_hyperparameters', {})
    search_space = {}
    final_selected = {}

    raw_models = config.get('models', [])
    if isinstance(raw_models, dict):
        for name, entry in raw_models.items():
            if isinstance(entry, dict):
                search_space[name] = deepcopy(entry.get('search_space') or entry.get('grid') or entry.get('params', {}))
                final_selected[name] = deepcopy(entry.get('params', {}))
    elif isinstance(raw_models, list):
        for entry in raw_models:
            if isinstance(entry, dict):
                name = entry.get('name') or entry.get('type') or 'model'
                search_space[name] = deepcopy(entry.get('search_space') or entry.get('grid') or entry.get('params', {}))
                final_selected[name] = deepcopy(entry.get('params', {}))
            elif isinstance(entry, str):
                final_selected[entry] = {}

    return {
        'model_names': model_names,
        'hyperparameters': hyperparameters,
        'search_space': search_space,
        'final_selected_configuration': final_selected,
    }


def build_experiment_manifest(
    metadata: Dict[str, Any],
    config: Dict[str, Any],
    artifacts: Dict[str, Any],
    project_root: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    environment = deepcopy(metadata.get('environment') or capture_environment())
    git_info = capture_git_info(project_root)
    timestamp = metadata.get('timestamp') or datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    model_section = _extract_model_section(metadata, config)
    dataset_fingerprint = metadata.get('dataset_hash') or metadata.get('fingerprint')
    if not dataset_fingerprint:
        dataset_source = metadata.get('dataset_source') or metadata.get('filepath')
        if dataset_source:
            try:
                dataset_fingerprint = compute_dataset_fingerprint(str(dataset_source))
            except Exception:
                dataset_fingerprint = None
    dataset = {
        'name': metadata.get('dataset_name'),
        'source': metadata.get('dataset_source'),
        'size': metadata.get('dataset_size'),
        'feature_count': metadata.get('feature_count'),
        'target_variable': metadata.get('target_column'),
        'class_distribution': metadata.get('class_distribution', {}),
        'fingerprint': dataset_fingerprint,
    }
    validation = {
        'random_seed': metadata.get('random_seed'),
        'validation_strategy': metadata.get('validation_strategy', {}),
        'cv_fold_count': metadata.get('validation_strategy', {}).get('n_splits'),
        'repetition_count': metadata.get('validation_strategy', {}).get('n_repetitions'),
        'train_test_split': {
            'test_size': config.get('dataset', {}).get('test_size'),
            'train_size': config.get('dataset', {}).get('train_size'),
        },
        'deterministic': bool(config.get('deterministic', False)),
        'persist_cv_splits': bool(config.get('persist_cv_splits', False)),
    }

    manifest = {
        'experiment_id': metadata.get('experiment_id'),
        'run_id': run_id or metadata.get('run_id') or f"run-{uuid4().hex}",
        'timestamp': timestamp,
        'framework_version': metadata.get('framework_version'),
        'git_commit': metadata.get('git_commit') or git_info.get('git_commit'),
        'dataset': dataset,
        'reproducibility': validation,
        'model': model_section,
        'environment': environment,
        'artifacts': deepcopy(artifacts),
        'metadata': deepcopy(metadata),
        'outputs': {
            'config_snapshot': deepcopy(config.get('_config_snapshot') or config),
            'config_checksum': config.get('_config_checksum'),
            'config_yaml_raw': config.get('_config_yaml_raw'),
        },
    }
    return manifest
