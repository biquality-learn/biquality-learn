import numpy as np
import pytest
from pytest import raises, warns
from sklearn.metrics.tests.test_classification import make_prediction
from sklearn.utils import shuffle

from bqlearn.metrics import (
    anchor_transition_matrix,
    gold_transition_matrix,
    iterative_anchor_transition_matrix,
)


def test_iterative_anchor_transition_matrix_equals_anchor_transition_matrix():
    _, _, probas_pred = make_prediction()

    assert np.allclose(
        iterative_anchor_transition_matrix(probas_pred, n_iter=1),
        anchor_transition_matrix(probas_pred),
    )


@pytest.mark.parametrize(
    "tm_estimator", [anchor_transition_matrix, iterative_anchor_transition_matrix]
)
def test_anchor_transition_matrix_shape(tm_estimator):
    _, _, probas_pred = make_prediction()

    n_classes = probas_pred.shape[1]

    tm = tm_estimator(probas_pred)

    assert tm.shape == (n_classes, n_classes)


def test_gold_transition_matrix_shape():
    y_true, _, probas_pred = make_prediction()

    n_classes = probas_pred.shape[1]

    tm = gold_transition_matrix(y_true, probas_pred)

    assert tm.shape == (n_classes, n_classes)


@pytest.mark.parametrize(
    "tm_estimator",
    [anchor_transition_matrix, iterative_anchor_transition_matrix],
)
@pytest.mark.parametrize(
    "quantile",
    [0.5, 0.9, 1.0],
)
def test_anchor_transition_matrix_row_sum_to_one(tm_estimator, quantile):
    _, _, probas_pred = make_prediction()

    n_classes = probas_pred.shape[1]

    tm = tm_estimator(probas_pred, quantile=quantile)

    assert np.allclose(np.sum(tm, axis=1), np.ones(n_classes))


def test_gold_transition_matrix_row_sum_to_one():
    y_true, _, probas_pred = make_prediction()

    n_classes = probas_pred.shape[1]

    tm = gold_transition_matrix(y_true, probas_pred)

    assert np.allclose(np.sum(tm, axis=1), np.ones(n_classes))


def test_gold_transition_matrix_string_labels_same_int_labels():
    y_true, _, probas_pred = make_prediction()

    y_labels = np.array(["Setosa", "Versicolour", "Virginica"])[y_true]

    tm1 = gold_transition_matrix(y_true, probas_pred)
    tm2 = gold_transition_matrix(
        y_labels, probas_pred, labels=["Setosa", "Versicolour", "Virginica"]
    )

    assert np.allclose(tm1, tm2)


def test_gold_transition_matrix_mix_labels():
    y_true, _, probas_pred = make_prediction()

    y_labels = np.array(["Setosa", "Versicolour", "Virginica"])[y_true]
    y_labels_mixed = np.array(["Versicolour", "Setosa", "Virginica"])[y_true]

    tm1 = gold_transition_matrix(y_labels_mixed, probas_pred)
    tm2 = gold_transition_matrix(y_labels, probas_pred)

    assert not np.allclose(tm1, tm2)


def test_gold_transition_matrix_absent_class():
    y_true, _, probas_pred = make_prediction()

    n_samples, n_classes = probas_pred.shape

    tm1 = gold_transition_matrix(y_true, probas_pred)

    labels = ["Setosa", "Versicolour", "Virginica", "ZZZ"]
    y_labels = np.array(labels)[y_true]
    probas_pred = np.c_[probas_pred, np.zeros(n_samples)]

    tm2 = gold_transition_matrix(y_labels, probas_pred, labels=labels)

    tm1_mod = np.c_[np.r_[tm1, np.zeros((1, n_classes))], np.zeros((n_classes + 1, 1))]
    tm1_mod[n_classes] = np.ones(n_classes + 1) / (n_classes + 1)

    assert np.allclose(tm1_mod, tm2)


def test_gold_transition_matrix_not_sorted_labels():
    y_true, _, probas_pred = make_prediction()

    labels = ["Versicolour", "Setosa", "Virginica"]
    y_labels = np.array(labels)[y_true]

    with warns(UserWarning):
        gold_transition_matrix(y_labels, probas_pred, labels=labels)


@pytest.mark.parametrize(
    "tm_estimator",
    [anchor_transition_matrix, iterative_anchor_transition_matrix],
)
def test_anchor_transition_matrix_shuffled(tm_estimator):
    _, _, probas_pred = make_prediction()

    tmp = np.copy(probas_pred)

    tm1 = tm_estimator(probas_pred)

    assert np.allclose(tmp, probas_pred)

    probas_pred = shuffle(probas_pred)

    tm2 = tm_estimator(probas_pred)

    assert np.allclose(tm1, tm2)


def test_gold_transition_matrix_shuffled():
    y_true, _, probas_pred = make_prediction()

    tm1 = gold_transition_matrix(y_true, probas_pred)

    y_true, probas_pred = shuffle(y_true, probas_pred)

    tm2 = gold_transition_matrix(y_true, probas_pred)

    assert np.allclose(tm1, tm2)


@pytest.mark.parametrize(
    "tm_estimator",
    [anchor_transition_matrix, iterative_anchor_transition_matrix],
)
def test_anchor_transition_matrix_no_proba(tm_estimator):
    non_probas = np.random.rand(100, 2)

    with raises(ValueError):
        tm_estimator(non_probas)

    with raises(ValueError):
        tm_estimator(10 * non_probas)

    with raises(ValueError):
        tm_estimator(-10 * non_probas)


def test_gold_transition_matrix_no_proba():
    non_probas = np.random.rand(100, 2)
    y = np.argmax(non_probas, axis=1)

    with raises(ValueError):
        gold_transition_matrix(y, non_probas)

    with raises(ValueError):
        gold_transition_matrix(y, 10 * non_probas)

    with raises(ValueError):
        gold_transition_matrix(y, -10 * non_probas)
