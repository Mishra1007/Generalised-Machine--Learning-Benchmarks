"""
Rerun a small set of previously failing datasets with safer preprocessing settings.
- Financial and merged use `encoding='ordinal'` to avoid one-hot expansion.
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from datasets import register_dataset
from validation.fold_manager import FoldManager
from sklearn.model_selection import KFold
from experiments.executor import ExperimentExecutor
from models.registry import list_models
from analysis.cbs_validation import run_cbs_validation

OUT_ROOT = PROJECT_ROOT / 'results'

RERUN = {
    'Edu-Primary': {'path': PROJECT_ROOT / 'datasets' / 'Edu-Primary.csv', 'target': 'G3', 'encoding': 'onehot'},
    'Financial': {'path': PROJECT_ROOT / 'datasets' / 'Financial.csv', 'target': 'Payment_Behaviour', 'encoding': 'ordinal'},
    'merged': {'path': PROJECT_ROOT / 'datasets' / 'merged.csv', 'target': None, 'encoding': 'ordinal'},
}


def detect_target_column(path: Path):
    try:
        import pandas as pd
        df = pd.read_csv(path, nrows=0)
        cols = list(df.columns)
        candidates = ['target', 'Outcome', 'Class', 'y', 'label', 'payment_behaviour', 'Payment_Behaviour', 'kredit', 'G3', 'Class']
        for c in candidates:
            if c in cols:
                return c
        if cols:
            return cols[-1]
    except Exception as e:
        logger.warning(f"Header detect failed for {path}: {e}")
    return None


def main():
    executor = ExperimentExecutor(seed=42)
    models = list_models()

    for name, info in RERUN.items():
        path = info['path']
        if not path.exists():
            logger.warning(f"Missing file for {name}: {path}")
            continue
        target = info.get('target') or detect_target_column(path)
        if not target:
            logger.warning(f"Could not detect target for {name}; skipping")
            continue
        register_dataset(name=name, filepath=str(path), target_column=target, description=f"Rerun {name}")

        # Adjust folds if dataset has very small class counts
        try:
            import pandas as pd
            df = pd.read_csv(path, usecols=[target])
            counts = df[target].value_counts()
            smallest = int(counts.min()) if not counts.empty else 0
            if smallest > 0 and smallest < executor.validator.n_splits:
                # If class singleton exists, fallback to non-stratified K-Fold
                if smallest < 2:
                    new_n = 2
                    class SimpleFoldManager:
                        def __init__(self, n_splits, n_repetitions, random_state):
                            self.n_splits = n_splits
                            self.n_repetitions = n_repetitions
                            self.random_state = random_state

                        def generate_folds(self, X, y):
                            for rep_id in range(self.n_repetitions):
                                kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state + rep_id)
                                fold_id = 0
                                for train_idx, test_idx in kf.split(X):
                                    yield rep_id, fold_id, train_idx, test_idx
                                    fold_id += 1

                        def validate_fold_indices(self, X, y):
                            if self.n_splits > len(y):
                                raise ValueError(f"Fold validation failed: n_splits={self.n_splits} greater than n_samples={len(y)}")
                            logger.info("✓ Fold indices validation passed (non-stratified)")
                            return True

                        def get_all_fold_info(self, X, y):
                            folds = []
                            for rep_id, fold_id, train_idx, test_idx in self.generate_folds(X, y):
                                folds.append({'repetition_id': rep_id, 'fold_id': fold_id, 'train_indices': train_idx.tolist(), 'test_indices': test_idx.tolist(), 'train_size': len(train_idx), 'test_size': len(test_idx)})
                            logger.info(f"Generated fold info for {len(folds)} folds total (non-stratified)")
                            return folds

                    executor.validator.n_splits = new_n
                    executor.validator.fold_manager = SimpleFoldManager(n_splits=new_n, n_repetitions=executor.validator.n_repetitions, random_state=executor.validator.random_state)
                    logger.info(f"Falling back to non-stratified KFold with n_splits={new_n} for dataset {name} (smallest class count={smallest})")
                else:
                    new_n = max(2, min(5, smallest))
                    executor.validator.n_splits = new_n
                    executor.validator.fold_manager = FoldManager(n_splits=new_n, n_repetitions=executor.validator.n_repetitions, random_state=executor.validator.random_state)
                    logger.info(f"Adjusted CV n_splits to {new_n} for dataset {name} (smallest class count={smallest})")
        except Exception:
            logger.debug("Could not adjust folds for dataset {name}")
        # If dataset is small or has many classes with singletons, disable stratification
        stratify = True
        if name == 'Edu-Primary':
            stratify = False

        cfg = {
            'dataset': {
                'registered_name': name,
                'test_size': 0.3,
                'preprocessing': {'scaling': 'standard', 'encoding': info.get('encoding', 'ordinal')},
                'stratify': stratify,
            },
            'models': [{'name': m, 'type': m} for m in models],
            'random_state': 42,
        }
        try:
            logger.info(f"Running benchmark for {name} (encoding={info.get('encoding')})")
            result = executor.run(cfg, out_root=str(OUT_ROOT))
            dataset_results_dir = OUT_ROOT / name
            logger.info(f"Running CBS validation for {name}")
            try:
                run_cbs_validation(str(dataset_results_dir), output_dir=str(dataset_results_dir / 'cbs_validation'), mc_iterations=2000, random_state=42)
            except Exception as e:
                logger.exception(f"CBS validation failed on rerun for {name}: {e}")
        except Exception:
            logger.exception(f"Rerun failed for {name}")

if __name__ == '__main__':
    main()
