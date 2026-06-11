"""Example: save benchmark results using the storage module."""
from metrics.storage import save_experiment_results


def main():
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

    config = {
        'task': 'binary_classification',
        'dataset_version': 'v1.2',
        'cv_folds': 5,
        'notes': 'Demo run for storage example'
    }

    class MockFoldResult:
        def __init__(self, rep, fold, acc):
            self.repetition_id = rep
            self.fold_id = fold
            self.metrics = {'accuracy': acc}

    class MockValResult:
        def __init__(self, accuracies):
            self.fold_results = [
                MockFoldResult(0, i, acc) for i, acc in enumerate(accuracies)
            ]

    # Dummy validation_results so significance testing triggers
    validation_results = {
        'LogisticRegression': MockValResult([0.86, 0.85, 0.87, 0.84, 0.86]),
        'RandomForest': MockValResult([0.91, 0.92, 0.90, 0.91, 0.93]),
        'SVM': MockValResult([0.87, 0.88, 0.86, 0.87, 0.89])
    }

    print('Saving experiment results for demo_dataset')
    saved = save_experiment_results(
        'demo_dataset_storage', 
        models, 
        out_root='results', 
        config=config, 
        seed=42,
        validation_results=validation_results
    )
    print('Saved:', saved)


if __name__ == '__main__':
    main()
