from itertools import islice

import numpy as np
import pytest
from sklearn import clone
from sklearn.datasets import make_classification
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from bqlearn.iter import iterative


@pytest.mark.parametrize(
    "estimator",
    [
        SGDClassifier(
            loss="log_loss",
            learning_rate="constant",
            eta0=0.1,
            shuffle=False,
            max_iter=10,
        ),
        LogisticRegression(solver="newton-cg", max_iter=10),
        MLPClassifier(
            hidden_layer_sizes=(),
            max_iter=10,
            learning_rate_init=0.00001,
            shuffle=False,
            batch_size=1000,
        ),
        GradientBoostingClassifier(n_estimators=10),
        HistGradientBoostingClassifier(max_iter=10),
        RandomForestClassifier(n_estimators=10),
    ],
)
def test_associativity_iterative_fit(estimator):
    seed = 1
    n_classes = 2
    n_samples = 1000

    X, y = make_classification(
        n_samples=n_samples,
        n_classes=n_classes,
        random_state=seed,
    )
    X = StandardScaler().fit_transform(X)

    estimator.set_params(random_state=seed)

    trained = clone(estimator).fit(X, y)
    iterated = next(
        islice(map(lambda e: e.fit(X, y), iterative(clone(estimator))), 9, None)
    )

    np.testing.assert_allclose(trained.predict_proba(X), iterated.predict_proba(X))
