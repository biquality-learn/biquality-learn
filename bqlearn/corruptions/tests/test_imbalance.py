import math

import numpy as np
import pytest
from sklearn.utils._testing import assert_array_equal

from .. import make_cluster_imbalance, make_imbalance


def test_make_imbalance():
    # Check that invalid arguments yield ValueError
    with pytest.raises(ValueError):
        make_imbalance([0], [0, 1])
    with pytest.raises(ValueError):
        make_imbalance([0, 1], [0, 1], majority_ratio=0)
    with pytest.raises(ValueError):
        make_imbalance([0, 1], [0, 1], imbalance_distribution="dumb")
    with pytest.raises(ValueError):
        make_imbalance([0, 1], [0, 1], minority_class_fraction=2)

    assert_array_equal(make_imbalance([0, 1], [0, 1]), ([0, 1], [0, 1]))


@pytest.mark.parametrize("majority_ratio", [2, 3, 10])
def test_subsample_imbalance(majority_ratio):
    rng = np.random.RandomState(1)
    n_samples = 1000
    y = rng.binomial(1, 0.9, size=n_samples)

    y_imbalanced = make_imbalance(
        y,
        majority_ratio=majority_ratio,
        imbalance_distribution="step",
        random_state=1,
    )

    minority_class = np.argmin(np.bincount(y))

    assert (
        math.floor(np.bincount(y)[minority_class] / majority_ratio)
        == np.bincount(y_imbalanced)[minority_class]
    )


@pytest.mark.parametrize("n_classes", [10, 100])
@pytest.mark.parametrize("majority_ratio", [2, 10])
def test_make_linear_imbalance(n_classes, majority_ratio):
    n_samples = 1000
    y = np.repeat(np.arange(0, n_classes), n_samples / n_classes)

    y_imbalanced = make_imbalance(
        y,
        majority_ratio=majority_ratio,
        imbalance_distribution="linear",
        random_state=0,
    )

    diff = np.diff(np.sort(np.bincount(y_imbalanced)))

    assert abs(np.max(diff) - np.min(diff)) <= 1


@pytest.mark.parametrize("n_classes", [10, 100])
@pytest.mark.parametrize("majority_ratio", [2, 10])
@pytest.mark.parametrize("minority_class_fraction", [0.2, 0.6])
def test_make_step_imbalance(n_classes, majority_ratio, minority_class_fraction):
    n_samples = 1000
    y = np.repeat(np.arange(0, n_classes), n_samples / n_classes)

    y_imbalanced = make_imbalance(
        y,
        majority_ratio=majority_ratio,
        imbalance_distribution="step",
        minority_class_fraction=minority_class_fraction,
        random_state=0,
    )

    imbalanced_class_ratio = np.bincount(y_imbalanced) / n_samples * n_classes
    n_minority_classes = round(n_classes * minority_class_fraction)
    idx_imbalanced_classes = imbalanced_class_ratio == 1 / majority_ratio

    assert np.count_nonzero(idx_imbalanced_classes) == n_minority_classes


def test_make_cluster_imbalance():
    # Check that invalid arguments yield ValueError
    with pytest.raises(ValueError):
        make_cluster_imbalance([[0]], [0, 1])
    with pytest.raises(ValueError):
        make_cluster_imbalance([[0], [1]], [0, 1], per_class_n_clusters=0)
    with pytest.raises(ValueError):
        make_cluster_imbalance([[0], [1]], [0, 1], per_class_n_clusters=["5"])

    assert make_cluster_imbalance([[0], [1]], [0, 1], per_class_n_clusters=1) == (
        [[0], [1]],
        [0, 1],
    )


@pytest.mark.parametrize("n_clusters", [2, 4])
@pytest.mark.parametrize("n_classes", [10, 20])
@pytest.mark.parametrize("majority_ratio", [2, 10])
@pytest.mark.parametrize("minority_class_fraction", [0.2, 0.6])
@pytest.mark.parametrize("imbalance_distribution", ["step", "linear"])
def test_cluster_imbalance_reproductible_parallel(
    n_clusters,
    n_classes,
    majority_ratio,
    minority_class_fraction,
    imbalance_distribution,
):
    rng = np.random.RandomState(0)
    n_samples = 1000
    X = rng.normal(size=(n_samples, 1))
    y = np.repeat(np.arange(0, n_classes), n_samples / n_classes)

    X_cluster_imbalanced, y_cluster_imbalanced = make_cluster_imbalance(
        X,
        y,
        per_class_n_clusters=n_clusters,
        majority_ratio=majority_ratio,
        minority_class_fraction=minority_class_fraction,
        imbalance_distribution=imbalance_distribution,
        random_state=0,
    )

    X_cluster_imbalanced_par, y_cluster_imbalanced_par = make_cluster_imbalance(
        X,
        y,
        per_class_n_clusters=n_clusters,
        majority_ratio=majority_ratio,
        minority_class_fraction=minority_class_fraction,
        imbalance_distribution=imbalance_distribution,
        random_state=0,
        n_jobs=-1,
    )

    assert np.allclose(X_cluster_imbalanced, X_cluster_imbalanced_par)
    assert np.allclose(y_cluster_imbalanced, y_cluster_imbalanced_par)
