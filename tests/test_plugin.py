import numpy as np
from pytest import raises
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.utils._testing import assert_array_equal

from bqlearn.plugin import PluginCorrection


def test_plugin_with_identity_matrix_equals_base_classifier():
    seed = 1
    n_classes = 2

    X, y = make_classification(
        n_samples=1000,
        n_classes=n_classes,
        random_state=seed,
    )

    base_clf = LogisticRegression(random_state=seed)
    plugin = PluginCorrection(clone(base_clf), transition_matrix=np.eye(n_classes))

    base_clf.fit(X, y)
    plugin.fit(X, y)

    assert_array_equal(base_clf.predict(X), plugin.predict(X))


def test_plugin_swap_predictions():
    seed = 1
    n_classes = 2

    X, y = make_classification(
        n_samples=1000,
        n_classes=n_classes,
        random_state=seed,
    )

    base_clf = LogisticRegression(random_state=seed)
    swapped_clf = PluginCorrection(
        clone(base_clf), transition_matrix=np.fliplr(np.eye(n_classes))
    )

    base_clf.fit(X, y)
    swapped_clf.fit(X, y)

    assert_array_equal(base_clf.predict(X), ~swapped_clf.predict(X) + 2)


def test_plugin_prefit_option_with_nonprefit_estimator_should_throw():
    seed = 1
    n_classes = 2

    X, y = make_classification(
        n_samples=1000,
        n_classes=n_classes,
        random_state=seed,
    )

    plugin = PluginCorrection(LogisticRegression(), prefit=True)

    with raises(NotFittedError):
        plugin.fit(X, y)
