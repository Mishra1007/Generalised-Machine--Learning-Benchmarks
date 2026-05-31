"""Integration smoke test: small dataset through runner->executor->storage.

This test is lightweight and intended to be runnable in CI.
It creates a tiny CSV, registers it, writes a temp YAML config, runs the runner,
and asserts that expected output files are produced.
"""
import os
import shutil
from pathlib import Path
import json
import tempfile

"""Note: this test imports the full stack (runner, executor, storage) and is not a unit test. It is intended as a smoke test to catch major integration issues. For more granular testing, see the unit tests in tests/test_metadata.py and tests/test_reproducibility_framework.py."""
import sys
from pathlib import Path

print("CWD:", Path.cwd())
print("SCRIPT:", Path(__file__).resolve())
print("SYS.PATH:")
for p in sys.path:
    print(" ", p)




import matplotlib
matplotlib.use('Agg')

from datasets.registry import register_dataset, get_dataset_config, get_registry
from experiments.runner import ExperimentRunner


def make_small_dataset(path: Path):
    # Create a tiny CSV with binary target
    import pandas as pd

    df = pd.DataFrame({
        'feat1': [0.1, 0.2, 0.2, 0.3, 0.9, 0.8],
        'feat2': [1, 2, 1, 2, 1, 2],
        'target': [0, 1, 0, 1, 1, 0]
    })
    df.to_csv(path, index=False)


def run_smoke_test():
    workspace = Path.cwd()
    tmpdir = Path(tempfile.mkdtemp(prefix='smoke_test_'))
    try:
        data_dir = tmpdir / 'data'
        data_dir.mkdir(parents=True)
        csv_path = data_dir / 'smoke.csv'
        make_small_dataset(csv_path)

        # Register dataset
        register_dataset('smoke_demo', str(csv_path), target_column='target', description='Smoke test dataset')

        # Write config YAML
        cfg_path = tmpdir / 'smoke_config.yaml'
        config = {
            'name': 'smoke_test',
            'dataset': {
                'registered_name': 'smoke_demo',
                'test_size': 0.33,
                'preprocessing': {'scaling': 'standard', 'encoding': 'onehot'}
            },
            'models': [
                {'name': 'lr', 'type': 'LogisticRegression', 'params': {'random_state': 42}},
            ],
            'random_state': 42,
        }

        import yaml
        with open(cfg_path, 'w', encoding='utf8') as fh:
            yaml.safe_dump(config, fh)

        # Output root
        out_root = str(tmpdir / 'results')

        # Run experiment
        runner = ExperimentRunner()
        result = runner.run_from_config(str(cfg_path), out_root=out_root)

        # Basic assertions
        saved = result.get('saved_files', {})
        assert saved, 'No saved files returned'
        assert os.path.exists(saved.get('raw')), 'Raw results not saved'
        assert os.path.exists(saved.get('normalized')), 'Normalized results not saved'
        assert os.path.exists(saved.get('cbs')), 'CBS results not saved'

        print('SMOKE TEST PASSED. Saved files:', json.dumps(saved, indent=2))

    finally:
        # Clean up
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass


if __name__ == '__main__':
    run_smoke_test()
