import numpy as np
import pytest
from scipy.stats import pearsonr
from sklearn.datasets import make_classification

from bqlearn.corruptions import make_weak_labels


@pytest.mark.parametrize("n_classes", [2, 10])
@pytest.mark.parametrize("n_samples", [1000])
@pytest.mark.parametrize("discrete", [True, False])
def test_make_weak_labels_empirical_quality_correlated_to_quality(
    n_classes, n_samples, discrete
):
    rng = np.random.RandomState(1)

    qs = np.linspace(0, 1, num=10, endpoint=True)

    X, y = make_classification(
        n_samples=n_samples,
        n_classes=n_classes,
        n_informative=n_classes,
        random_state=rng,
    )

    emp_qs = []

    for q in qs:
        y_corrupted = make_weak_labels(
            X,
            y,
            train_size=q,
            discrete=discrete,
            random_state=rng,
        )

        emp_q = 1 - np.sum(y != y_corrupted) / n_samples

        emp_qs.append(emp_q)

    assert pearsonr(qs, emp_qs)[0] > 0
