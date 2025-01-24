import math
from functools import partial

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from bqlearn.corruptions import (
    make_feature_dependent_label_noise,
    make_instance_dependent_label_noise,
    make_label_noise,
    noisy_leaves_probability,
    uncertainty_noise_probability,
)
from bqlearn.corruptions.noise_matrices import (
    background_noise_matrix,
    flip_noise_matrix,
    uniform_noise_matrix,
)


@pytest.mark.parametrize("noise_ratio", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("n_classes", [2, 10, 100])
@pytest.mark.parametrize("n_samples", [10000])
@pytest.mark.parametrize(
    "noise_matrix",
    [
        uniform_noise_matrix,
        flip_noise_matrix,
        background_noise_matrix,
        partial(flip_noise_matrix, permutation=True),
    ],
)
def test_make_label_noise_reproduces_noise_matrix(
    noise_ratio, n_classes, n_samples, noise_matrix
):
    rng = np.random.RandomState(1)

    y = rng.randint(n_classes, size=n_samples)

    matrix = noise_matrix(n_classes, noise_ratio)

    y_corrupted = make_label_noise(y, matrix)

    estimated_matrix = confusion_matrix(y, y_corrupted, normalize="true")

    assert np.allclose(estimated_matrix, matrix, atol=0.1, rtol=0.2)


@pytest.mark.parametrize(
    "noise_matrix", ["uniform", "flip", "background", "permutation"]
)
def test_make_label_noise_seedable(noise_matrix):
    seed = 1
    noise_ratio = 0.5
    n_classes = 10
    n_samples = 10000

    rng = np.random.RandomState(seed)

    y = rng.randint(n_classes, size=n_samples)

    y_corrupted = make_label_noise(
        y, noise_matrix, noise_ratio=noise_ratio, random_state=seed
    )
    y_corrupted_2 = make_label_noise(
        y, noise_matrix, noise_ratio=noise_ratio, random_state=seed
    )

    assert np.allclose(y_corrupted, y_corrupted_2)


@pytest.mark.parametrize("noise_ratio", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("n_classes", [2, 10])
@pytest.mark.parametrize("n_samples", [100])
@pytest.mark.parametrize(
    "noise_matrix", ["uniform", "flip", "background", "permutation"]
)
def test_mln_equals_midln_with_const_noise(
    noise_ratio, n_classes, n_samples, noise_matrix
):
    seed = 1
    rng = np.random.RandomState(seed)
    y = rng.randint(n_classes, size=n_samples)

    y_corrupted = make_label_noise(
        y, noise_matrix, noise_ratio=noise_ratio, random_state=seed
    )
    y_instance_corrupted = make_instance_dependent_label_noise(
        noise_ratio * np.ones_like(y), y, noise_matrix, random_state=seed
    )

    assert np.array_equal(y_corrupted, y_instance_corrupted)


@pytest.mark.parametrize("noise_ratio", [0.0, 0.2, 0.5, 0.8, 1.0])
@pytest.mark.parametrize("purity", ["random", "ascending", "descending"])
@pytest.mark.parametrize("n_classes", [2, 10])
@pytest.mark.parametrize("n_samples", [100])
def test_noise_ratio_for_noisy_leaves(noise_ratio, purity, n_classes, n_samples):
    rng = np.random.RandomState(0)
    n_samples = 1000
    X = rng.normal(size=(n_samples, 1))
    y = np.repeat(np.arange(0, n_classes), n_samples / n_classes)

    noise_prob = noisy_leaves_probability(X, y, noise_ratio=noise_ratio, purity=purity)

    assert math.isclose(np.mean(noise_prob), noise_ratio, abs_tol=0.02)


@pytest.mark.parametrize("noise_ratio", [0.0, 0.2, 0.5, 0.8, 1.0])
@pytest.mark.parametrize("n_classes", [2, 10])
@pytest.mark.parametrize("uncertainty", ["entropy", "uncertainty", "margin"])
def test_noise_ratio_for_uncertainty(noise_ratio, n_classes, uncertainty):
    rng = np.random.RandomState(0)
    n_samples = 100
    X = rng.normal(size=(n_samples, 1))
    y = np.repeat(np.arange(0, n_classes), n_samples / n_classes)

    clf = LogisticRegression()
    clf.fit(X, y)

    noise_prob = uncertainty_noise_probability(
        X, clf, uncertainty=uncertainty, noise_ratio=noise_ratio
    )

    assert math.isclose(np.mean(noise_prob), noise_ratio)


@pytest.mark.parametrize("noise_ratio", [0.0, 0.2, 0.5, 0.8, 1.0])
@pytest.mark.parametrize("n_classes", [2, 10])
def test_noise_ratio_for_feature_noise(noise_ratio, n_classes):
    rng = np.random.RandomState(0)
    n_samples = 1000
    X = rng.normal(size=(n_samples, 1))
    y = np.repeat(np.arange(0, n_classes), n_samples / n_classes)

    y_noisy = make_feature_dependent_label_noise(
        X, y, noise_ratio=noise_ratio, random_state=0
    )

    assert math.isclose(np.mean(y_noisy != y), noise_ratio, abs_tol=0.1)
