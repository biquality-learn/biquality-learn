import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

from .. import IKMM, IPDR


@pytest.mark.parametrize(
    "idr",
    [
        IPDR(DecisionTreeClassifier(), n_estimators=1),
        IKMM(DecisionTreeClassifier(), n_estimators=1, kernel="linear"),
    ],
)
def test_idr_exploit_warning(idr):
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

    idr.set_params(exploit_iterative_learning=True)
    with pytest.raises(ValueError):
        idr.fit(X, y, sample_quality=sample_quality)

    idr.set_params(exploit_iterative_learning=False)
    assert idr.fit(X, y, sample_quality=sample_quality) is not None
