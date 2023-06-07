import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from .. import KKMM, KPDR


@pytest.mark.parametrize("n_classes", [10, 20])
def test_kdr_missing_class_in_untrusted(n_classes):
    rng = np.random.RandomState(0)
    n_samples = 1000
    X = rng.normal(size=(n_samples, 1))
    y = np.repeat(np.arange(0, n_classes), n_samples / n_classes)
    only_class_untrusted = rng.choice(range(n_classes))
    sample_quality = np.ones(1000)
    sample_quality[
        (y == only_class_untrusted) & (rng.randint(0, 2, size=n_samples).astype(bool))
    ] = 0

    clf = LogisticRegression()

    KPDR(clf, method="odds").fit(X, y, sample_quality)
    KKMM(clf, kernel="linear").fit(X, y, sample_quality)
