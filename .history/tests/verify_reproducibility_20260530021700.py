import tempfile
import json
from pathlib import Path
import shutil

from experiments.metadata import compute_dataset_fingerprint
from datasets.registry import register_dataset
from experiments.executor import ExperimentExecutor
from validation.fold_manager import FoldManager
import metrics.storage as storage


def write_csv(path: Path, text: str):
    path.write_text(text)


def run_checks():
    report = {
        'passed': [],
        'failed': [],
        'edge_cases': [],
    }

    tmp = Path(tempfile.mkdtemp(prefix='verify_'))
    try:
        # 1. Same dataset -> identical hash
        csv1 = tmp / 'd1.csv'
        write_csv(csv1, 'a,b,target\n1,2,0\n3,4,1\n')
        h1 = compute_dataset_fingerprint(str(csv1))
        h1b = compute_dataset_fingerprint(str(csv1))
        if h1 == h1b:
            report['passed'].append('same-dataset-hash')
        else:
            report['failed'].append('same-dataset-hash')

        # 2. Different content changes hash
        csv2 = tmp / 'd2.csv'
        write_csv(csv2, 'a,b,target\n9,8,0\n7,6,1\n')
        h2 = compute_dataset_fingerprint(str(csv2))
        if h1 != h2:
            report['passed'].append('different-dataset-hash')
        else:
            report['failed'].append('different-dataset-hash')

        # 3-7: Run an experiment and inspect results
        data_dir = tmp / 'data'
        data_dir.mkdir()
        csv = data_dir / 'sm.csv'
        write_csv(csv, 'feat1,feat2,target\n0.1,1,0\n0.2,2,1\n0.3,1,0\n0.4,2,1\n')
        register_dataset('verify_demo', str(csv), target_column='target')

        cfg = {
            'dataset': {'registered_name': 'verify_demo', 'test_size': 0.5, 'preprocessing': {'scaling': 'standard'}},
            'models': [{'name': 'lr', 'type': 'LogisticRegression', 'params': {'random_state': 42}}],
            'random_state': 42,
        }

        out_root = str(tmp / 'results')
        exe = ExperimentExecutor(seed=42)
        res = exe.run(cfg, out_root=out_root)

        base = Path(out_root) / 'verify_demo'

        # 3. Config snapshot matches execution config (experiment metadata stored)
        # metadata.json should exist
        meta_path = base / 'metadata.json'
        if meta_path.exists():
            report['passed'].append('metadata_saved')
            meta = json.loads(meta_path.read_text(encoding='utf8'))
            # config snapshot == cfg (subset check)
            cfg_snap = meta.get('config_snapshot', {})
            # Compare keys present in cfg
            mismatch = False
            for k in cfg:
                if cfg.get(k) != cfg_snap.get(k):
                    mismatch = True
                    break
            if not mismatch:
                report['passed'].append('config_snapshot_match')
            else:
                report['failed'].append('config_snapshot_match')
        else:
            report['failed'].append('metadata_saved')

        # 5. Missing metadata cannot occur silently -> ensure metadata.json exists always
        if not meta_path.exists():
            report['failed'].append('metadata_presence')
        else:
            report['passed'].append('metadata_presence')

        # 6. CV split metadata reconstructs original folds
        folds = meta.get('folds', [])
        if folds:
            # Build FoldManager and check one fold
            fm = FoldManager(n_splits=exe.validator.n_splits, n_repetitions=exe.validator.n_repetitions, random_state=exe.validator.random_state)
            # load data for reconstruction
            from datasets.loaders import DatasetLoader
            loader = DatasetLoader()
            X, y, _ = loader.load_csv(str(csv), 'target')
            # Compare first fold indices
            first = folds[0]
            rep_id = first['repetition_id']
            fold_id = first['fold_id']
            info = fm.get_fold_info(X.values, y.values, rep_id, fold_id)
            if list(info['test_indices']) == list(first['test_indices']):
                report['passed'].append('cv_split_reconstruct')
            else:
                report['failed'].append('cv_split_reconstruct')
        else:
            report['edge_cases'].append('no_cv_folds_saved')

        # 7. Result bundle completeness
        expected = ['config.json', 'metadata.json', 'metrics.json', 'raw_results.csv', 'normalized_results.csv', 'cbs_scores.csv']
        missing = [p for p in expected if not (base / p).exists()]
        if not missing:
            report['passed'].append('result_bundle_complete')
        else:
            report['failed'].append({'result_bundle_complete': missing})

        # 8. Backward compatibility: run with minimal config (no persist flags) and ensure no crash
        cfg2 = {
            'dataset': {'registered_name': 'verify_demo'},
            'models': ['lr'],
        }
        try:
            res2 = exe.run(cfg2, out_root=str(tmp / 'results2'))
            report['passed'].append('backward_compatible')
        except Exception as e:
            report['failed'].append({'backward_compatible': str(e)})

        # Reproducibility score heuristic
        score = 0
        score += 2 if 'same-dataset-hash' in report['passed'] else 0
        score += 2 if 'different-dataset-hash' in report['passed'] else 0
        score += 2 if 'config_snapshot_match' in report['passed'] else 0
        score += 2 if 'metadata_presence' in report['passed'] else 0
        score += 2 if 'cv_split_reconstruct' in report['passed'] else 0
        report['reproducibility_score'] = score

    finally:
        try:
            shutil.rmtree(tmp)
        except Exception:
            pass

    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    run_checks()
