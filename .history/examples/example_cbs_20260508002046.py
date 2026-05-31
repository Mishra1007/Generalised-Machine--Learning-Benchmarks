"""Example demonstrating normalization and CBS ranking."""

from metrics import (
    normalize_model_summaries,
    compute_cbs,
    rank_models_by_cbs,
)

# Simulated per-model summaries (as would come from MetricsCalculator.get_comprehensive_summary)
models = {
    'LogisticRegression': {
        'overall': {
            'f1_mean': 0.85,
            'roc_auc_mean': 0.89,
            'accuracy_mean': 0.86,
            'precision_mean': 0.84,
            'recall_mean': 0.86,
            'total_time_mean': 0.05,
        },
        'stability': {
            'by_metric': {'f1': {'stability_score': 0.95}},
            'assessment': {'primary_metric_cv': 0.02}
        }
    },
    'RandomForest': {
        'overall': {
            'f1_mean': 0.92,
            'roc_auc_mean': 0.94,
            'accuracy_mean': 0.91,
            'precision_mean': 0.90,
            'recall_mean': 0.92,
            'total_time_mean': 0.5,
        },
        'stability': {
            'by_metric': {'f1': {'stability_score': 0.90}},
            'assessment': {'primary_metric_cv': 0.03}
        }
    },
    'SVM': {
        'overall': {
            'f1_mean': 0.88,
            'roc_auc_mean': 0.90,
            'accuracy_mean': 0.87,
            'precision_mean': 0.86,
            'recall_mean': 0.88,
            'total_time_mean': 2.0,
        },
        'stability': {
            'by_metric': {'f1': {'stability_score': 0.92}},
            'assessment': {'primary_metric_cv': 0.025}
        }
    }
}

print('Normalizing metrics (per-dataset min-max)...')
normalized = normalize_model_summaries(models, primary_metric='f1')
for name, vals in normalized.items():
    print(f"- {name}: {vals}")

print('\nComputing CBS...')
cbs = compute_cbs(models, primary_metric='f1')
for name, info in cbs.items():
    print(f"- {name}: CBS={info['cbs']:.4f}, normalized={info['normalized']}")

print('\nRanking models by CBS:')
ranked = rank_models_by_cbs(cbs)
for i, (name, score) in enumerate(ranked, start=1):
    print(f"{i}. {name}: {score:.4f}")
