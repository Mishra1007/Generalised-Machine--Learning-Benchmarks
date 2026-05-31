"""Example demonstrating the visualization module for benchmarking.

Generates publication-quality plots and saves them under `plots/<dataset>/`.
"""
from metrics.visualization import generate_plots_for_dataset


def main():
    # Simulated model summaries (reuse structure used by previous examples)
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

    print('Generating plots for dataset: demo_dataset')
    paths = generate_plots_for_dataset('demo_dataset', models, out_dir='plots')
    for name, (png, pdf) in paths.items():
        print(f"Saved {name}: {png}, {pdf}")


if __name__ == '__main__':
    main()
