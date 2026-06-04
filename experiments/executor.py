"""Experiment executor: runs a single experiment given a parsed config.

This module implements the core execution flow and performs dependency
injection for the dataset/preprocessing/validation/metrics/storage steps.
"""
from typing import Dict, Any, Optional
import logging
import uuid
import time
from copy import deepcopy
from pathlib import Path

import random
import numpy as np
import datetime

from preprocessing.data_preparation import DataPreparation, prepare_dataset
from models.registry import create_model, list_models
from validation.cross_validator import CrossValidator
from metrics.pipeline import MetricsCalculator
from metrics import normalization as metrics_normalization
from metrics import cbs as metrics_cbs
from metrics import storage as metrics_storage
from experiments import metadata as metadata_mod
import sys
from reproducibility import FRAMEWORK_VERSION, capture_environment
from reproducibility.environment import capture_git_info

logger = logging.getLogger(__name__)


def _json_safe(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


class ExperimentExecutor:
    """Run a single experiment based on config and provided components.

    The executor is independent of config parsing and is suitable for unit tests.
    """

    def __init__(self, seed: Optional[int] = 42):
        self.seed = seed or 42
        self.validator = CrossValidator(random_state=self.seed)
        self.metrics_calc = MetricsCalculator()

    def _set_seeds(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)

    def run(self, config: Dict[str, Any], out_root: str = 'results') -> Dict[str, Any]:
        """Execute experiment flow described in config.

        Returns a dict containing model summaries and saved file paths.
        """
        experiment_id = config.get('experiment_id') or f"exp-{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
        run_id = config.get('run_id') or f"run-{uuid.uuid4().hex[:10]}"
        config['experiment_id'] = experiment_id
        config['run_id'] = run_id

        seed = config.get('random_state', self.seed)
        self._set_seeds(seed)

        dataset_cfg = config.get('dataset', {})
        dataset_name = dataset_cfg.get('registered_name') or dataset_cfg.get('name') or dataset_cfg.get('dataset_name') or 'dataset'
        dataset_source = dataset_cfg.get('source') or dataset_cfg.get('filepath')

        ds_name = dataset_cfg.get('registered_name') or dataset_cfg.get('name', 'iris')
        X_train, X_test, y_train, y_test, dataset_meta = prepare_dataset(
            ds_name,
            test_size=dataset_cfg.get('test_size', 0.3),
            random_state=seed,
            scaling_method=dataset_cfg.get('preprocessing', {}).get('scaling', 'standard'),
            encoding_method=dataset_cfg.get('preprocessing', {}).get('encoding', 'onehot'),
            stratify=dataset_cfg.get('stratify', True),
        )
        dataset_name = ds_name
        X_train_array = np.asarray(X_train)
        y_train_array = np.asarray(y_train)

        logger.info(f"Dataset prepared: {dataset_name} — train {X_train.shape[0]}, test {X_test.shape[0]}")

        models_cfg = config.get('models', [])
        models_map = {}
        deterministic = bool(config.get('deterministic', False))
        if isinstance(models_cfg, dict):
            for name, args in models_cfg.items():
                try:
                    params = deepcopy(args.get('params', {}) or {}) if isinstance(args, dict) else {}
                    if deterministic and 'random_state' not in params:
                        params['random_state'] = seed
                    models_map[name] = create_model(args.get('type', name), **params)
                except Exception:
                    models_map[name] = create_model(name)
        elif isinstance(models_cfg, list):
            for entry in models_cfg:
                if isinstance(entry, str):
                    models_map[entry] = create_model(entry, random_state=seed) if deterministic else create_model(entry)
                elif isinstance(entry, dict):
                    name = entry.get('name') or entry.get('type')
                    params = deepcopy(entry.get('params', {}) or {})
                    if deterministic and 'random_state' not in params:
                        params['random_state'] = seed
                    models_map[name] = create_model(entry.get('type', name), **params)

        logger.info(f"Initialized {len(models_map)} models: {list(models_map.keys())}")

        persist_cv = bool(config.get('persist_cv_splits', True))
        save_predictions = bool(config.get('save_predictions', False))

        # Preflight CV validation to avoid running models with invalid fold settings
        try:
            self.validator.fold_manager.validate_fold_indices(X_train_array, y_train_array)
        except Exception as exc:
            logger.error(f"CV validation failed before model execution: {exc}")
            raise

        try:
            folds_info = self.validator.fold_manager.get_all_fold_info(X_train_array, y_train_array)
        except Exception:
            try:
                from validation.fold_manager import FoldManager
                fm = FoldManager(n_splits=self.validator.n_splits, n_repetitions=self.validator.n_repetitions, random_state=self.validator.random_state)
                folds_info = fm.get_all_fold_info(X_train_array, y_train_array)
            except Exception:
                folds_info = []

        results_map = {}
        for model_name, model in models_map.items():
            try:
                results_map[model_name] = self.validator.validate(
                    X_train_array,
                    y_train_array,
                    model,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    predict_proba=True,
                    return_predictions=save_predictions,
                )
            except Exception:
                results_map[model_name] = None

        model_summaries = {}
        for model_name, vresults in results_map.items():
            if vresults is None:
                logger.warning(f"No results for {model_name}")
                continue

            fold_list = []
            for fr in vresults.fold_results:
                fold_metrics = dict(fr.metrics)
                fold_metrics['train_time'] = getattr(fr, 'train_time', 0.0)
                fold_metrics['eval_time'] = getattr(fr, 'eval_time', 0.0)
                fold_list.append(fold_metrics)

            model_summaries[model_name] = self.metrics_calc.get_comprehensive_summary(
                fold_list,
                model_name=model_name,
                dataset_name=dataset_name,
            )

        cbs_map = metrics_cbs.compute_cbs(model_summaries)

        model_names = list(models_map.keys())
        model_hps = {}
        cfg_models = config.get('models', [])
        if isinstance(cfg_models, dict):
            for k, v in cfg_models.items():
                model_hps[k] = deepcopy(v.get('params', {}) if isinstance(v, dict) else {})
        elif isinstance(cfg_models, list):
            for entry in cfg_models:
                if isinstance(entry, dict):
                    name = entry.get('name') or entry.get('type')
                    model_hps[name] = deepcopy(entry.get('params', {}) or {})

        project_root = str(Path(__file__).resolve().parents[1])
        framework_version = str(config.get('framework_version') or FRAMEWORK_VERSION)
        git_commit = capture_git_info(project_root).get('git_commit')
        dataset_source_path = dataset_meta.get('dataset_source') or dataset_meta.get('filepath') or dataset_source
        dataset_hash = dataset_meta.get('dataset_hash') or dataset_meta.get('fingerprint')
        if not dataset_hash and dataset_source_path:
            try:
                dataset_hash = metadata_mod.compute_dataset_fingerprint(str(dataset_source_path), target_column=dataset_meta.get('target_name'))
            except Exception:
                dataset_hash = None
        env_info = capture_environment()
        serialized_folds = [_json_safe(fold) for fold in folds_info]

        exp_meta = metadata_mod.ExperimentMetadata(
            experiment_id=experiment_id,
            run_id=run_id,
            timestamp=datetime.datetime.utcnow().isoformat() + 'Z',
            framework_version=framework_version,
            git_commit=git_commit,
            dataset_name=dataset_name,
            dataset_source=dataset_source_path,
            dataset_hash=dataset_hash,
            dataset_size=int(dataset_meta.get('n_samples', 0)),
            feature_count=int(dataset_meta.get('feature_count', dataset_meta.get('n_features', 0))),
            target_column=dataset_meta.get('target_name') or dataset_meta.get('target_column'),
            class_distribution=dataset_meta.get('class_distribution', {}),
            preprocessing_pipeline=deepcopy(config.get('dataset', {}).get('preprocessing', {})),
            model_names=model_names,
            model_hyperparameters=model_hps,
            validation_strategy={
                'n_splits': self.validator.n_splits,
                'n_repetitions': self.validator.n_repetitions,
                'train_test_split': {
                    'test_size': dataset_cfg.get('test_size', 0.3),
                    'train_size': dataset_cfg.get('train_size'),
                },
            },
            folds=serialized_folds if persist_cv else [],
            repetitions=self.validator.n_repetitions,
            random_seed=seed,
            config_snapshot=config.get('_config_snapshot') or deepcopy(config),
            python_version=env_info.get('python_version'),
            library_versions=env_info.get('library_versions'),
            environment=env_info,
        )

        saved = metrics_storage.save_experiment_results(
            dataset_name,
            model_summaries,
            out_root=out_root,
            config=config,
            seed=seed,
            metadata=exp_meta.to_dict(),
            validation_results=results_map if save_predictions else None,
        )

        return {
            'experiment_id': experiment_id,
            'run_id': run_id,
            'dataset': dataset_name,
            'model_summaries': model_summaries,
            'cbs_map': cbs_map,
            'saved_files': saved,
        }
