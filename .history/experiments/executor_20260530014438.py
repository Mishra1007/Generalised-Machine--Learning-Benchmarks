"""Experiment executor: runs a single experiment given a parsed config.

This module implements the core execution flow and performs dependency
injection for the dataset/preprocessing/validation/metrics/storage steps.
"""
from typing import Dict, Any, Optional
import logging
import uuid
import time

import random
import numpy as np

from preprocessing.data_preparation import DataPreparation, prepare_dataset
from models.registry import create_model, list_models
from validation.cross_validator import CrossValidator
from metrics.pipeline import MetricsCalculator
from metrics import normalization as metrics_normalization
from metrics import cbs as metrics_cbs
from metrics import storage as metrics_storage

logger = logging.getLogger(__name__)


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
        seed = config.get('random_state', self.seed)
        self._set_seeds(seed)

        dataset_cfg = config.get('dataset', {})
        dataset_name = dataset_cfg.get('name') or dataset_cfg.get('dataset_name') or 'dataset'

        # 1) Prepare data (train/test). Use convenience API when registry name is provided.
        if isinstance(dataset_cfg, dict) and dataset_cfg.get('registered_name'):
            prep = DataPreparation(random_state=seed)
            X_train, X_test, y_train, y_test, metadata = prep.prepare_dataset(
                dataset_cfg['registered_name'],
                test_size=dataset_cfg.get('test_size', 0.3),
                random_state=seed,
                scaling_method=dataset_cfg.get('preprocessing', {}).get('scaling', 'standard'),
                encoding_method=dataset_cfg.get('preprocessing', {}).get('encoding', 'onehot'),
            )
            dataset_name = dataset_cfg['registered_name']
        else:
            # Fallback: expect prepare_dataset(dataset_name,..) or dataset_name directly
            prep = DataPreparation(random_state=seed)
            X_train, X_test, y_train, y_test, metadata = prepare_dataset(
                dataset_cfg.get('name', 'iris'),
                test_size=dataset_cfg.get('test_size', 0.3),
                random_state=seed,
                scaling_method=dataset_cfg.get('preprocessing', {}).get('scaling', 'standard'),
                encoding_method=dataset_cfg.get('preprocessing', {}).get('encoding', 'onehot'),
            )

        logger.info(f"Dataset prepared: {dataset_name} — train {X_train.shape[0]}, test {X_test.shape[0]}")

        # 2) Initialize models from config
        models_cfg = config.get('models', [])
        models_map = {}
        if isinstance(models_cfg, dict):
            # mapping name -> args
            for name, args in models_cfg.items():
                try:
                    models_map[name] = create_model(args.get('type', name), **(args.get('params', {}) or {}))
                except Exception:
                    # fallback: try create_model on name
                    models_map[name] = create_model(name)
        elif isinstance(models_cfg, list):
            for entry in models_cfg:
                if isinstance(entry, str):
                    models_map[entry] = create_model(entry)
                elif isinstance(entry, dict):
                    name = entry.get('name') or entry.get('type')
                    params = entry.get('params', {})
                    models_map[name] = create_model(entry.get('type', name), **params)

        logger.info(f"Initialized {len(models_map)} models: {list(models_map.keys())}")

        # 3) Run validation on training data (ensures fairness)
        results_map = self.validator.validate_multiple(X_train, y_train, models_map, dataset_name=dataset_name)

        # 4) Build model summaries
        model_summaries = {}
        for model_name, vresults in results_map.items():
            if vresults is None:
                logger.warning(f"No results for {model_name}")
                continue

            fold_list = []
            for fr in vresults.fold_results:
                # fr.metrics already contains predictive & some computational metrics
                m = dict(fr.metrics)
                m['train_time'] = getattr(fr, 'train_time', 0.0)
                m['eval_time'] = getattr(fr, 'eval_time', 0.0)
                fold_list.append(m)

            summary = self.metrics_calc.get_comprehensive_summary(fold_list, model_name=model_name, dataset_name=dataset_name)
            model_summaries[model_name] = summary

        # 5) Compute CBS (normalized inside compute_cbs)
        cbs_map = metrics_cbs.compute_cbs(model_summaries)

        # 6) Generate visualizations and save outputs
        saved = metrics_storage.save_experiment_results(dataset_name, model_summaries, out_root=out_root, config=config, seed=seed)

        return {
            'experiment_id': config.get('experiment_id'),
            'dataset': dataset_name,
            'model_summaries': model_summaries,
            'cbs_map': cbs_map,
            'saved_files': saved,
        }
