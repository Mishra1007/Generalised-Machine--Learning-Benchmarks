import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from validation.results import ValidationResults, FoldResult
from metrics.storage import _build_score_matrix
from analysis.significance import global_significance_analysis
from analysis.statistics import rank_models


def _make_validation_results():
    # Two models with two folds and identical fold keys
    results = {}
    for model_name, scores in {
        'model_a': [0.8, 0.9],
        'model_b': [0.6, 0.7],
    }.items():
        vres = ValidationResults(model_name=model_name, dataset_name='demo', random_state=42)
        for fold_id, score in enumerate(scores):
            fold = FoldResult(
                repetition_id=0,
                fold_id=fold_id,
                model_name=model_name,
                dataset_name='demo',
                metrics={'accuracy': score},
                train_size=10,
                test_size=5,
            )
            vres.add_fold_result(fold)
        results[model_name] = vres
    return results


def test_build_score_matrix_shape_and_alignment():
    validation_results = _make_validation_results()
    matrix, model_names = _build_score_matrix(validation_results, primary_metric='accuracy')
    assert model_names == ['model_a', 'model_b']

    mat = np.asarray(matrix, dtype=float)
    # Expected shape: observations (folds) x models
    assert mat.shape == (2, 2)
    assert np.allclose(mat[:, 0], [0.8, 0.9])
    assert np.allclose(mat[:, 1], [0.6, 0.7])


def test_significance_and_ranking_use_columns_as_models():
    validation_results = _make_validation_results()
    matrix, model_names = _build_score_matrix(validation_results, primary_metric='accuracy')
    mat = np.asarray(matrix, dtype=float)

    analysis = global_significance_analysis(mat, model_names=model_names)
    ranking = analysis['ranking_table']
    assert ranking.iloc[0]['model'] == 'model_a'

    rank_df = rank_models(mat, model_names=model_names)
    assert rank_df.iloc[0]['model'] == 'model_a'


if __name__ == '__main__':
    test_build_score_matrix_shape_and_alignment()
    test_significance_and_ranking_use_columns_as_models()
    print('Matrix orientation tests passed')
