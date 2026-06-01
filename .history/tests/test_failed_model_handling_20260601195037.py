import sys
from pathlib import Path
import tempfile
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.base import BaseModel
from models.registry import register_model
from experiments.executor import ExperimentExecutor
from validation.cross_validator import CrossValidator
from datasets.registry import register_dataset


class _ExplodingEstimator:
    def fit(self, X, y):
        raise ValueError('intentional failure')


class ExplodingModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(name='ExplodingModel', **kwargs)

    def _build_model(self):
        return _ExplodingEstimator()


def _write_dataset(path: Path):
    rows = []
    for i in range(20):
        rows.append({'feat1': i * 0.1, 'feat2': i % 3, 'target': i % 2})
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def test_failed_model_training_does_not_crash_visualization():
    register_model('ExplodingModel', ExplodingModel)

    with tempfile.TemporaryDirectory(prefix='fail_model_') as tmp:
        tmp_path = Path(tmp)
        csv_path = tmp_path / 'data.csv'
        _write_dataset(csv_path)

        register_dataset('fail_model_demo', str(csv_path), target_column='target')

        config = {
            'dataset': {
                'registered_name': 'fail_model_demo',
                'test_size': 0.2,
                'preprocessing': {'scaling': 'standard', 'encoding': 'onehot'},
            },
            'models': [{'name': 'explode', 'type': 'ExplodingModel'}],
            'random_state': 42,
        }

        executor = ExperimentExecutor(seed=42)
        executor.validator = CrossValidator(n_splits=2, n_repetitions=1, random_state=42)

        result = executor.run(config, out_root=str(tmp_path / 'results'))
        saved = result.get('saved_files', {})

        assert saved and Path(saved['raw']).exists()
        raw_lines = Path(saved['raw']).read_text(encoding='utf8').strip().splitlines()
        assert len(raw_lines) >= 1


if __name__ == '__main__':
    test_failed_model_training_does_not_crash_visualization()
    print('Failed model handling test passed')
