import numpy as np
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.utils._testing import assert_array_almost_equal, assert_array_equal

from bqlearn.unbiased import LossCorrection


def test_unbiaised_swap_predictions():
    seed = 1
    n_classes = 2

    X, y = make_classification(
        n_samples=1000,
        n_classes=n_classes,
        random_state=seed,
    )

    base_clf = LogisticRegression(random_state=seed)
    swapped_clf = LossCorrection(
        clone(base_clf), transition_matrix=np.fliplr(np.eye(n_classes))
    )

    base_clf.fit(X, y)
    swapped_clf.fit(X, y)

    assert_array_equal(base_clf.predict(X), ~swapped_clf.predict(X) + 2)
    assert_array_almost_equal(
        base_clf.decision_function(X), -swapped_clf.decision_function(X)
    )


def test_unbiaised_with_identity_matrix_equals_base_classifier():
    seed = 1
    n_classes = 2

    X, y = make_classification(
        n_samples=1000,
        n_classes=n_classes,
        random_state=seed,
    )

    base_clf = LogisticRegression(random_state=seed)
    swapped_clf = LossCorrection(clone(base_clf), transition_matrix=np.eye(n_classes))

    base_clf.fit(X, y)
    swapped_clf.fit(X, y)

    assert_array_equal(base_clf.predict(X), swapped_clf.predict(X))
    assert_array_almost_equal(
        base_clf.decision_function(X), swapped_clf.decision_function(X)
    )
