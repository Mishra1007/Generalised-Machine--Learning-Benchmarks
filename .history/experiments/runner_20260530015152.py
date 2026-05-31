"""Experiment runner: parses config, generates experiment ID, and delegates to executor.

Provides a small CLI to run experiments with a YAML config file. Uses dependency
injection to keep components testable and preserves existing architecture.
"""
import argparse
import yaml
import logging
import datetime
import uuid
from pathlib import Path

from experiments.executor import ExperimentExecutor

logger = logging.getLogger(__name__)


class ConfigValidationError(ValueError):
    pass


class ExperimentRunner:
    def __init__(self, executor: ExperimentExecutor = None):
        self.executor = executor or ExperimentExecutor()

    def _load_config(self, path: str) -> dict:
        with open(path, 'r', encoding='utf8') as fh:
            cfg = yaml.safe_load(fh)
        return cfg or {}

    def _validate_config(self, cfg: dict) -> None:
        """Basic YAML schema validation. Fail fast with clear messages.

        Rules (minimal required):
        - top-level `dataset` must be present and contain either `registered_name` or `name`.
        - top-level `models` must be present and be a non-empty list or dict.
        - `random_state` if present must be an integer.
        """
        if not isinstance(cfg, dict):
            raise ConfigValidationError("Experiment config must be a mapping (YAML dict)")

        if 'dataset' not in cfg:
            raise ConfigValidationError("Missing required 'dataset' section in config")

        dataset = cfg['dataset']
        if not isinstance(dataset, dict):
            raise ConfigValidationError("'dataset' must be a mapping with 'registered_name' or 'name'")

        if not (bool(dataset.get('registered_name')) or bool(dataset.get('name'))):
            raise ConfigValidationError("'dataset' must include either 'registered_name' or 'name'")

        if 'models' not in cfg:
            raise ConfigValidationError("Missing required 'models' section in config")

        models = cfg['models']
        if not models:
            raise ConfigValidationError("'models' must be a non-empty list or mapping")

        if 'random_state' in cfg and not isinstance(cfg['random_state'], int):
            raise ConfigValidationError("'random_state' must be an integer if provided")

    def run_from_config(self, path: str, out_root: str = 'results') -> dict:
        cfg = self._load_config(path)

        # Create experiment ID and add to config for provenance
        experiment_id = cfg.get('experiment_id') or f"exp-{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
        cfg['experiment_id'] = experiment_id

        # Ensure random_state present
        if 'random_state' not in cfg:
            cfg['random_state'] = cfg.get('seed', 42)

        logger.info(f"Running experiment {experiment_id} from {path}")
        result = self.executor.run(cfg, out_root=out_root)
        logger.info(f"Experiment {experiment_id} complete. Saved files: {result.get('saved_files')}")
        return result


def _cli():
    parser = argparse.ArgumentParser(description='Run ML benchmark experiment from YAML config')
    parser.add_argument('--config', '-c', required=True, help='Path to experiment YAML config')
    parser.add_argument('--out', '-o', default='results', help='Output root directory')
    args = parser.parse_args()

    runner = ExperimentRunner()
    runner.run_from_config(args.config, out_root=args.out)


if __name__ == '__main__':
    _cli()
