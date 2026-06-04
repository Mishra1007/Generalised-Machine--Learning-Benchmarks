import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.effect_size import effect_size_summary
from analysis.significance import nemenyi_test, pairwise_wilcoxon
from analysis.statistics import friedman_test, rank_models
from metrics.storage import _build_score_matrix
from validation.results import FoldResult, ValidationResults


def _fold(model_name, repetition_id, fold_id, score):
    return FoldResult(
        repetition_id=repetition_id,
        fold_id=fold_id,
        model_name=model_name,
        dataset_name='synthetic_orientation',
        metrics={'accuracy': score},
        train_size=20,
        test_size=5,
    )


def _make_validation_results(score_by_model):
    results = {}
    for model_name, keyed_scores in score_by_model.items():
        validation = ValidationResults(model_name=model_name, dataset_name='synthetic_orientation', random_state=7)
        for repetition_id, fold_id, score in keyed_scores:
            validation.add_fold_result(_fold(model_name, repetition_id, fold_id, score))
        results[model_name] = validation
    return results


def _synthetic_matrix():
    # Rows are aligned observations; columns are models A, B, C.
    return np.array(
        [
            [0.90, 0.80, 0.70],
            [0.85, 0.86, 0.70],
            [0.95, 0.75, 0.77],
            [0.88, 0.88, 0.79],
        ],
        dtype=float,
    )


def test_build_score_matrix_aligns_models_by_sorted_fold_keys():
    validation_results = _make_validation_results(
        {
            'model_a': [(0, 2, 0.95), (0, 0, 0.90), (0, 1, 0.85), (0, 3, 0.88)],
            'model_b': [(0, 3, 0.88), (0, 1, 0.86), (0, 0, 0.80), (0, 2, 0.75)],
            'model_c': [(0, 1, 0.70), (0, 0, 0.70), (0, 3, 0.79), (0, 2, 0.77), (0, 99, 0.01)],
        }
    )

    matrix, model_names = _build_score_matrix(validation_results, primary_metric='accuracy')

    assert model_names == ['model_a', 'model_b', 'model_c']
    assert np.allclose(np.asarray(matrix, dtype=float), _synthetic_matrix())


def test_build_score_matrix_rejects_duplicate_observation_keys():
    validation_results = _make_validation_results(
        {
            'model_a': [(0, 0, 0.90), (0, 0, 0.91)],
            'model_b': [(0, 0, 0.80), (0, 1, 0.81)],
        }
    )

    with pytest.raises(ValueError, match='Duplicate fold observation key'):
        _build_score_matrix(validation_results, primary_metric='accuracy')


def test_friedman_input_shape_uses_rows_as_observations_and_columns_as_models():
    matrix = _synthetic_matrix()

    expected = stats.friedmanchisquare(matrix[:, 0], matrix[:, 1], matrix[:, 2])
    actual = friedman_test(matrix, model_names=['model_a', 'model_b', 'model_c'])

    assert math.isclose(actual['statistic'], expected.statistic, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(actual['p_value'], expected.pvalue, rel_tol=1e-12, abs_tol=1e-12)

    with pytest.raises(ValueError, match='Model names must match the number of columns'):
        friedman_test(matrix.T, model_names=['model_a', 'model_b', 'model_c'])


def test_wilcoxon_paired_alignment_uses_identical_observation_rows():
    matrix = _synthetic_matrix()

    pairwise = pairwise_wilcoxon(matrix, model_names=['model_a', 'model_b', 'model_c'])
    actual = pairwise.loc[(pairwise['model_a'] == 'model_a') & (pairwise['model_b'] == 'model_b')].iloc[0]
    expected = stats.wilcoxon(matrix[:, 0], matrix[:, 1], zero_method='wilcox', alternative='two-sided', mode='auto')

    assert math.isclose(actual['statistic'], expected.statistic, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(actual['p_value_raw'], expected.pvalue, rel_tol=1e-12, abs_tol=1e-12)


def test_nemenyi_rank_generation_ranks_each_observation_row():
    matrix = _synthetic_matrix()

    result = nemenyi_test(matrix, model_names=['model_a', 'model_b', 'model_c'], higher_is_better=True)

    expected_rank_rows = np.array([stats.rankdata(-row, method='average') for row in matrix], dtype=float)
    expected_average_ranks = expected_rank_rows.mean(axis=0)

    assert np.allclose(
        [result['average_ranks']['model_a'], result['average_ranks']['model_b'], result['average_ranks']['model_c']],
        expected_average_ranks,
        rtol=1e-12,
        atol=1e-12,
    )
    assert result['ranking_table'].iloc[0]['model'] == 'model_a'


def test_rank_models_aggregates_over_observation_rows():
    matrix = _synthetic_matrix()

    ranking = rank_models(matrix, model_names=['model_a', 'model_b', 'model_c'])

    assert np.allclose(ranking.sort_values('model')['mean_score'], matrix.mean(axis=0), rtol=1e-12, atol=1e-12)
    assert ranking.iloc[0]['model'] == 'model_a'


def test_effect_size_alignment_uses_paired_differences():
    matrix = _synthetic_matrix()
    diffs = matrix[:, 0] - matrix[:, 1]

    summary = effect_size_summary(matrix[:, 0], matrix[:, 1], paired=True)
    expected_d = diffs.mean() / diffs.std(ddof=1)

    assert math.isclose(summary['cohens_d'], expected_d, rel_tol=1e-12, abs_tol=1e-12)

    with pytest.raises(ValueError, match='paired samples of equal length'):
        effect_size_summary(matrix[:, 0], matrix[:-1, 1], paired=True)
