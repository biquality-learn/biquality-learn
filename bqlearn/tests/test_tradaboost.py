import math

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import fetch_openml, make_blobs, make_classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import zero_one_loss
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from bqlearn.tradaboost import TrAdaBoostClassifier


def test_replicate_edible_vs_poisonous():
    # Fetch mushroom from openml
    X, y = fetch_openml(data_id=24, return_X_y=True, as_frame=True, parser="liac-arff")

    mask = X["stalk-shape"] == "e"

    total = np.arange(X.shape[0])
    diff = total[~mask]

    X.drop("stalk-shape", axis=1, inplace=True)

    X = OneHotEncoder().fit_transform(X)
    y = LabelEncoder().fit_transform(y)

    n_rounds = 10

    svm_errs = []
    svmt_errs = []
    tradaboost_errs = []

    for iround in range(n_rounds):
        same, test = train_test_split(
            total[mask],
            train_size=int(0.01 * diff.shape[0]),
            random_state=iround,
        )
        train = np.hstack((same, diff))

        base_svm = SGDClassifier(
            loss="hinge", class_weight="balanced", alpha=1e-8, random_state=iround
        )

        svm = clone(base_svm)
        svm.fit(X[same], y[same])
        svm_err = zero_one_loss(y[test], svm.predict(X[test]))
        svm_errs.append(svm_err)

        svmt = clone(base_svm)
        svmt.fit(X[train], y[train])
        svmt_err = zero_one_loss(y[test], svmt.predict(X[test]))
        svmt_errs.append(svmt_err)

        tradaboost = TrAdaBoostClassifier(
            estimator=clone(base_svm),
            n_estimators=100,
            learning_rate=1.0,
            random_state=iround,
        )
        tradaboost.fit(
            X[train],
            y[train],
            sample_quality=mask[train],
        )
        tradaboost_err = zero_one_loss(y[test], tradaboost.predict(X[test]))
        tradaboost_errs.append(tradaboost_err)

    svm_err = np.mean(svm_errs)
    svmt_err = np.mean(svmt_errs)
    tradaboost_err = np.mean(tradaboost_errs)

    assert svm_err < 0.127
    assert svmt_err < 0.135
    assert tradaboost_err < 0.071
    assert tradaboost_err < svm_err
    assert tradaboost_err < svmt_err


def test_tradaboost_recovers_adaboost():
    rng = np.random.RandomState(1)

    X, y = make_classification(
        n_samples=1000,
        n_classes=2,
        random_state=rng,
    )

    base_clf = DecisionTreeClassifier()

    adaboost = AdaBoostClassifier(base_clf, algorithm="SAMME")
    tradaboost = TrAdaBoostClassifier(base_clf)

    adaboost.fit(X, y)
    tradaboost.fit(X, y, sample_quality=np.ones_like(y))

    assert np.allclose(adaboost.decision_function(X), tradaboost.decision_function(X))


@pytest.mark.parametrize("n_classes", [2, 3, 5, 10])
@pytest.mark.parametrize("learning_rate", [0.01, 0.1, 0.5, 0.1])
def test_weight_drift_correction_tradaboost(n_classes, learning_rate):
    n_samples = 1000

    seed = 0

    X, y = make_blobs(n_samples=n_samples, centers=n_classes, random_state=seed)

    trusted, untrusted = next(
        StratifiedShuffleSplit(train_size=0.1, random_state=seed).split(X, y)
    )
    sample_quality = np.ones(n_samples)
    sample_quality[untrusted] = 0

    trada = TrAdaBoostClassifier(
        DecisionTreeClassifier(max_depth=None), learning_rate=learning_rate
    )
    sample_weight = np.ones(n_samples)
    sample_weights = []
    max_iter = 20
    for i in range(1, max_iter + 1):
        trada.set_params(n_estimators=i)
        trada.fit(X, y, sample_quality=sample_quality)
        sample_weight /= sample_weight.sum()
        sample_weights.append(sample_weight)
        sample_weight, _, _ = trada._boost(
            i + 1, X, y, np.copy(sample_weight), sample_quality, seed, {}
        )

    sample_weights = np.stack(sample_weights)
    sum_sw_trusted = np.sum(sample_weights[:, trusted], axis=1)
    sum_sw_untrusted = np.sum(sample_weights[:, untrusted], axis=1)

    assert math.isclose(np.min(sum_sw_trusted), np.max(sum_sw_trusted))
    assert math.isclose(np.min(sum_sw_untrusted), np.max(sum_sw_untrusted))
