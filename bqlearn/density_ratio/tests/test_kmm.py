import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from .._kmm import IKMM, KKMM


@pytest.mark.parametrize("batch_size", [-1, 0, 1.2])
@pytest.mark.parametrize(
    "estimator", [KKMM(LogisticRegression()), IKMM(LogisticRegression())]
)
def test_wrong_batch_size_raises_value_error(batch_size, estimator):
    seed = 1
    n_classes = 2
    n_samples = 1000

    rng = np.random.RandomState(seed)

    X, y = make_classification(
        n_samples=n_samples,
        n_classes=n_classes,
        random_state=seed,
    )
    sample_quality = rng.randint(0, 2, size=n_samples)

    estimator.set_params(batch_size=batch_size)

    with pytest.raises(ValueError):
        estimator.fit(X, y, sample_quality)
