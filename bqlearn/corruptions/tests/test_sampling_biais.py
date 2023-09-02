import math

import numpy as np

from bqlearn.corruptions import make_sampling_biais


def test_sampling_biais_small_a():
    rng = np.random.RandomState(0)
    n_samples = 100
    X = rng.normal(size=(n_samples, 1))

    X_biaised = make_sampling_biais(X, a=0.9, random_state=0)

    assert np.mean(X_biaised) < np.mean(X)


def test_sampling_biais_big_a():
    rng = np.random.RandomState(0)
    n_samples = 100
    X = rng.normal(size=(n_samples, 1))

    X_biaised = make_sampling_biais(X, a=1.1, random_state=0)

    assert np.mean(X_biaised) > np.mean(X)


def test_sampling_biais_a_equals_one():
    rng = np.random.RandomState(0)
    n_samples = 10000
    X = rng.normal(size=(n_samples, 1))

    X_biaised = make_sampling_biais(X, a=1, random_state=0)

    assert math.isclose(np.mean(X_biaised), np.mean(X), abs_tol=0.02)
