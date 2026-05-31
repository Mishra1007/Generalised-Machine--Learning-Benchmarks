import json
from pathlib import Path

import numpy as np

from datasets.registry import register_dataset
from experiments.runner import ExperimentRunner
from reproducibility import capture_environment
from validation.fold_manager import FoldManager


def _write_dataset(path: Path, rows: int = 60) -> None:
    lines = ['feat1,feat2,target']
    for i in range(rows):
        lines.append(f'{i * 0.1},{i % 3},{i % 2}')
    path.write_text('\n'.join(lines) + '\n', encoding='utf8')


def _register_demo_dataset(name: str, csv_path: Path) -> None:
    register_dataset(name, str(csv_path), target_column='target', description='reproducibility test dataset')


def test_identical_seeds_produce_identical_splits(tmp_path):
    csv = tmp_path / 'splits.csv'
    _write_dataset(csv)
    _register_demo_dataset('repro_splits', csv)

    from preprocessing.data_preparation import prepare_dataset

    X_train_1, X_test_1, y_train_1, y_test_1, _ = prepare_dataset('repro_splits', test_size=0.3, random_state=7)
    X_train_2, X_test_2, y_train_2, y_test_2, _ = prepare_dataset('repro_splits', test_size=0.3, random_state=7)

    fm_1 = FoldManager(n_splits=5, n_repetitions=3, random_state=7)
    fm_2 = FoldManager(n_splits=5, n_repetitions=3, random_state=7)

    folds_1 = fm_1.get_all_fold_info(np.asarray(X_train_1), np.asarray(y_train_1))
    folds_2 = fm_2.get_all_fold_info(np.asarray(X_train_2), np.asarray(y_train_2))

    def normalize(folds):
        return [
            {
                **fold,
                'train_indices': list(fold['train_indices']),
                'test_indices': list(fold['test_indices']),
            }
            for fold in folds
        ]

    assert np.array_equal(np.asarray(X_train_1), np.asarray(X_train_2))
    assert np.array_equal(np.asarray(y_train_1), np.asarray(y_train_2))
    assert normalize(folds_1) == normalize(folds_2)


def test_manifests_are_generated_correctly(tmp_path):
    csv = tmp_path / 'manifest.csv'
    _write_dataset(csv)
    _register_demo_dataset('repro_manifest', csv)

    cfg = {
        'experiment_id': 'exp-test-manifest',
        'run_id': 'run-test-manifest',
        'dataset': {
            'registered_name': 'repro_manifest',
            'test_size': 0.25,
            'preprocessing': {'scaling': 'standard', 'encoding': 'onehot'},
        },
        'models': [{'name': 'lr', 'type': 'LogisticRegression', 'params': {'random_state': 7}}],
        'random_state': 7,
        'deterministic': True,
        'persist_cv_splits': True,
    }

    cfg_path = tmp_path / 'config.yaml'
    cfg_path.write_text(
        json.dumps(cfg),
        encoding='utf8',
    )

    runner = ExperimentRunner()
    result = runner.run_from_config(str(cfg_path), out_root=str(tmp_path / 'results'))
    base = tmp_path / 'results' / 'repro_manifest'

    manifest_path = base / 'experiment_manifest.json'
    report_path = base / 'reproducibility_report.md'

    assert manifest_path.exists()
    assert report_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    assert manifest['experiment_id'] == 'exp-test-manifest'
    assert manifest['run_id'] == 'run-test-manifest'
    assert manifest['dataset']['name'] == 'repro_manifest'
    assert manifest['dataset']['fingerprint']
    assert manifest['environment']['packages']['numpy']
    assert manifest['environment']['packages']['scipy']
    assert manifest['environment']['launcher_environment']['PYTHONHASHSEED'] is not None
    assert manifest['artifacts']['config'].endswith('config.json')
    assert manifest['artifacts']['cbs_scores'].endswith('cbs_scores.csv')


def test_environment_capture_succeeds():
    env = capture_environment()
    for key in ['python_version', 'os', 'cpu', 'packages', 'thread_environment']:
        assert key in env
    for package_name in ['numpy', 'pandas', 'scikit-learn', 'scipy', 'joblib']:
        assert package_name in env['packages']
    assert 'launcher_environment' in env


def test_metadata_persists_across_runs(tmp_path):
    csv = tmp_path / 'persist.csv'
    _write_dataset(csv)
    _register_demo_dataset('repro_persist', csv)

    cfg = {
        'dataset': {'registered_name': 'repro_persist', 'test_size': 0.25},
        'models': ['lr'],
        'random_state': 11,
        'deterministic': True,
        'persist_cv_splits': True,
    }

    runner = ExperimentRunner()
    first = runner.executor.run(cfg.copy(), out_root=str(tmp_path / 'run1'))
    second = runner.executor.run(cfg.copy(), out_root=str(tmp_path / 'run2'))

    first_manifest = json.loads((tmp_path / 'run1' / 'repro_persist' / 'experiment_manifest.json').read_text(encoding='utf8'))
    second_manifest = json.loads((tmp_path / 'run2' / 'repro_persist' / 'experiment_manifest.json').read_text(encoding='utf8'))

    stable_first = {k: v for k, v in first_manifest.items() if k not in {'experiment_id', 'run_id', 'timestamp'}}
    stable_second = {k: v for k, v in second_manifest.items() if k not in {'experiment_id', 'run_id', 'timestamp'}}

    assert stable_first == stable_second
    assert first['saved_files']['manifest']
    assert second['saved_files']['manifest']
