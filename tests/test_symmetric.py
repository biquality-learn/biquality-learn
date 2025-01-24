import numpy as np
import pytest
from pytest import raises
from sklearn.datasets import make_classification

from bqlearn.unhinged import KernelUnhinged, LinearUnhinged


@pytest.mark.parametrize("alpha", np.power(10, np.arange(-5, 5, dtype=float)))
def test_linear_unhinged_regularization_coef_scaling(alpha):
    rng = np.random.RandomState(1)

    X, y = make_classification(
        n_samples=1000,
        n_classes=2,
        random_state=rng,
    )

    unhinged = LinearUnhinged()
    unhinged_reg = LinearUnhinged(alpha=alpha)

    unhinged.fit(X, y)
    unhinged_reg.fit(X, y)

    assert np.allclose(unhinged.coef_ * (1 / alpha), unhinged_reg.coef_)


@pytest.mark.parametrize("clf", [KernelUnhinged(), LinearUnhinged()])
def test_unhinged_is_binary_classifier(clf):
    rng = np.random.RandomState(1)

    X, y = make_classification(
        n_samples=1000,
        n_classes=3,
        n_informative=4,
        random_state=rng,
    )

    with raises(ValueError):
        clf.fit(X, y)


def test_kernel_unhinged_with_linear_kernel_equals_linear_unhinged():
    rng = np.random.RandomState(1)

    X, y = make_classification(
        n_samples=1000,
        n_classes=2,
        random_state=rng,
    )

    sample_weight = rng.normal(size=1000)

    kernel_unhinged = KernelUnhinged(alpha=1 / 2, kernel="linear")
    linear_unhinged = LinearUnhinged(alpha=1 / 2)

    kernel_unhinged.fit(X, y, sample_weight=sample_weight)
    linear_unhinged.fit(X, y, sample_weight=sample_weight)

    assert np.allclose(
        linear_unhinged.decision_function(X), kernel_unhinged.decision_function(X)
    )
