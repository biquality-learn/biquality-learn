import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils._testing import assert_array_almost_equal, assert_array_equal

from bqlearn.multiclass import WeightedOneVsRestClassifier


@pytest.mark.parametrize("n_classes", [2, 10])
def test_weighted_ovr_with_noweights_equals_ovr(n_classes):
    seed = 1

    X, y = make_classification(
        n_samples=1000,
        n_classes=n_classes,
        n_informative=n_classes,
        random_state=seed,
    )

    base = LogisticRegression(multi_class="ovr", random_state=seed)
    ovr = OneVsRestClassifier(LogisticRegression(random_state=seed))
    wovr = WeightedOneVsRestClassifier(LogisticRegression(random_state=seed))

    base.fit(X, y)
    ovr.fit(X, y)
    wovr.fit(X, y)

    assert_array_equal(base.predict(X), ovr.predict(X))
    assert_array_equal(ovr.predict(X), wovr.predict(X))
    assert_array_almost_equal(base.predict_proba(X), ovr.predict_proba(X))
    assert_array_almost_equal(ovr.predict_proba(X), wovr.predict_proba(X))
    assert_array_almost_equal(base.decision_function(X), ovr.decision_function(X))
    assert_array_almost_equal(ovr.decision_function(X), wovr.decision_function(X))
