"""
Run cross-domain CBS validation across listed datasets.

This script will:
- Register datasets found in `datasets/` with reasonable target column guesses
- Run the standard model suite on each dataset using `ExperimentExecutor`
- Run `run_cbs_validation` on each results/<dataset>/ folder
- Aggregate key CBS validation outputs and write `CBS_CROSS_DOMAIN_VALIDATION.md`

Usage: python scripts/run_cross_domain_validation.py
"""

import logging
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from datasets import register_dataset, get_registry
from experiments.executor import ExperimentExecutor
from models.registry import list_models
from analysis.cbs_validation import run_cbs_validation

# Dataset -> guessed target column
DATASETS = {
    'Edu-Primary': PROJECT_ROOT / 'datasets' / 'Edu-Primary.csv',
    'Edu-xApi': PROJECT_ROOT / 'datasets' / 'Edu-xApi.csv',
    'Financial': PROJECT_ROOT / 'datasets' / 'Financial.csv',
    'german_credit_data': PROJECT_ROOT / 'datasets' / 'german_credit_data.csv',
    'diabetes': PROJECT_ROOT / 'datasets' / 'diabetes.csv',
    'heart': PROJECT_ROOT / 'datasets' / 'heart.csv',
    'merged': PROJECT_ROOT / 'datasets' / 'merged.csv',
}

# Initial guesses for target columns (may be overridden by header inspection)
TARGET_GUESSES = {
    'Edu-Primary': 'G3',
    'Edu-xApi': 'Class',
    'Financial': 'Payment_Behaviour',
    'german_credit_data': 'kredit',
    'diabetes': 'Outcome',
    'heart': 'target',
    'merged': None,  # detect
}

OUT_ROOT = PROJECT_ROOT / 'results'


def detect_target_column(path: Path):
    try:
        import pandas as pd
        df = pd.read_csv(path, nrows=0)
        cols = list(df.columns)
        # Heuristics: prefer common names
        candidates = ['target', 'Outcome', 'Class', 'y', 'label', 'payment_behaviour', 'Payment_Behaviour', 'kredit', 'G3']
        for c in candidates:
            if c in cols:
                return c
        # fallback: last column
        if cols:
            return cols[-1]
    except Exception as e:
        logger.warning(f"Failed to inspect {path}: {e}")
    return None


def main():
    executor = ExperimentExecutor(seed=42)
    models = list_models()
    logger.info(f"Models to run: {models}")

    summaries = []

    for name, path in DATASETS.items():
        path = Path(path)
        if not path.exists():
            logger.warning(f"Dataset file not found: {path} — skipping {name}")
            continue

        target = TARGET_GUESSES.get(name)
        if target is None:
            target = detect_target_column(path)
        if not target:
            logger.warning(f"Could not determine target for {name} — skipping")
            continue

        logger.info(f"Registering dataset {name} -> {path} (target={target})")
        register_dataset(name=name, filepath=str(path), target_column=target, description=f"Auto-registered {name}")

        cfg = {
            'dataset': {'registered_name': name, 'test_size': 0.3, 'preprocessing': {'scaling': 'standard', 'encoding': 'onehot'}},
            'models': [{'name': m, 'type': m} for m in models],
            'random_state': 42,
        }

        try:
            logger.info(f"Running benchmark for {name}")
            result = executor.run(cfg, out_root=str(OUT_ROOT))
            saved = result.get('saved_files') or result.get('saved_files', {})
            dataset_results_dir = OUT_ROOT / name

            # Run CBS validation on the saved results
            logger.info(f"Running CBS validation for {name}")
            try:
                cbs_out = run_cbs_validation(str(dataset_results_dir), output_dir=str(dataset_results_dir / 'cbs_validation'), mc_iterations=2000, random_state=42)
            except Exception as e:
                logger.exception(f"CBS validation failed for {name}: {e}")
                cbs_out = None

            summaries.append({'dataset': name, 'results_dir': str(dataset_results_dir), 'cbs_artifacts': cbs_out})
        except Exception:
            logger.exception(f"Benchmark failed for {name}")
            continue

    # Aggregate cross-domain results
    logger.info("Aggregating cross-domain CBS validation outputs")
    rows = []
    for s in summaries:
        name = s['dataset']
        base = Path(s['results_dir']) / 'cbs_validation'
        if not base.exists():
            logger.warning(f"No cbs_validation output for {name}")
            continue
        # read ranking_stability and normalization_validation if present
        rs_path = base / 'ranking_stability.csv'
        norm_path = base / 'normalization_validation.csv'
        wr_path = base / 'weight_robustness.csv'
        mi_path = base / 'metric_influence.csv'

        entry = {'dataset': name}
        try:
            if rs_path.exists():
                rs = pd.read_csv(rs_path)
                entry['top1_prob_mean'] = float(rs['top1_probability'].mean()) if 'top1_probability' in rs.columns else None
                entry['cbs_std_mean'] = float(rs['cbs_std'].mean()) if 'cbs_std' in rs.columns else None
            if norm_path.exists():
                nv = pd.read_csv(norm_path)
                entry['compression_ratio_mean'] = float(nv['compression_ratio'].mean()) if 'compression_ratio' in nv.columns else None
                entry['rank_corr_mean'] = float(nv['rank_correlation'].mean()) if 'rank_correlation' in nv.columns else None
            if wr_path.exists():
                wr = pd.read_csv(wr_path)
                entry['weight_sensitivity_mean'] = float(wr['spearman_corr'].mean()) if 'spearman_corr' in wr.columns else None
            if mi_path.exists():
                mi = pd.read_csv(mi_path)
                # metric influence: top influencing metric per model — take average of absolute influence
                if 'influence' in mi.columns:
                    entry['metric_influence_mean_abs'] = float(mi['influence'].abs().mean())
        except Exception as e:
            logger.warning(f"Failed to aggregate for {name}: {e}")

        rows.append(entry)

    agg_df = pd.DataFrame(rows)
    report_path = PROJECT_ROOT / 'project_artifacts/audits/CBS_CROSS_DOMAIN_VALIDATION.md'
    with open(report_path, 'w', encoding='utf8') as fh:
        fh.write('# CBS Cross-Domain Validation Report\n\n')
        fh.write('Generated by scripts/run_cross_domain_validation.py\n\n')
        fh.write('## Per-dataset summary\n\n')
        if not agg_df.empty:
            fh.write(agg_df.to_markdown(index=False))
            fh.write('\n\n')
        else:
            fh.write('No aggregated results found.\n')

        fh.write('## Notes\n\n')
        fh.write('- This report summarizes a set of automated CBS validation runs.\n')
        fh.write('- Investigate per-dataset `results/<dataset>/cbs_validation` for full artifacts.\n')
        fh.write('\n')
        fh.write('## Threats to validity\n\n')
        fh.write('- Automated target-column detection may be incorrect for some datasets.\n')
        fh.write('- Model suite is the default registered classification models; some datasets may be regression in nature.\n')
        fh.write('- CBS weights were not modified as requested.\n')

    logger.info(f"Wrote aggregated report to {report_path}")


if __name__ == '__main__':
    main()
