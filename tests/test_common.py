import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin, clone
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.utils._test_common.instance_generator import (
    PER_ESTIMATOR_XFAIL_CHECKS as SK_PER_ESTIMATOR_XFAIL_CHECKS,
)
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import _num_samples, check_is_fitted

from bqlearn.baseline import BiqualityBaseline
from bqlearn.density_ratio import IKMM, IPDR, KKMM, KPDR
from bqlearn.ea import EasyADAPT
from bqlearn.irbl import IRBL
from bqlearn.irlnl import IRLNL
from bqlearn.multiclass import WeightedOneVsRestClassifier
from bqlearn.plugin import PluginCorrection
from bqlearn.tradaboost import TrAdaBoostClassifier
from bqlearn.unbiased import LossCorrection
from bqlearn.unhinged import KernelUnhinged, LinearUnhinged


@parametrize_with_checks(
    [
        LinearUnhinged(),
        KernelUnhinged(),
        WeightedOneVsRestClassifier(SGDClassifier(random_state=0), n_jobs=-1),
    ],
    expected_failed_checks=lambda estimator: {
        LinearUnhinged: {
            "check_sample_weight_equivalence_on_dense_data": (
                "see issue https://github.com/scikit-learn/scikit-learn/issues/30770"
            ),
            "check_sample_weight_equivalence_on_sparse_data": (
                "see issue https://github.com/scikit-learn/scikit-learn/issues/30770"
            ),
        },
        KernelUnhinged: {
            "check_sample_weight_equivalence_on_dense_data": (
                "see issue https://github.com/scikit-learn/scikit-learn/issues/30770"
            ),
            "check_sample_weight_equivalence_on_sparse_data": (
                "see issue https://github.com/scikit-learn/scikit-learn/issues/30770"
            ),
        },
        WeightedOneVsRestClassifier: {
            "check_sample_weight_equivalence_on_dense_data": (
                "sample_weight is not equivalent to removing/repeating samples."
            ),
            "check_sample_weight_equivalence_on_sparse_data": (
                "sample_weight is not equivalent to removing/repeating samples."
            ),
            "check_sample_weights_shape": "per class sample weights with ndim=2",
        },
    }.get(type(estimator), {}),
)
def test_all_estimators(estimator, check):
    return check(estimator)


@parametrize_with_checks([EasyADAPT()])
def test_all_transformers(estimator, check):
    return check(estimator)


@parametrize_with_checks(
    [
        BiqualityBaseline(LogisticRegression(), baseline="no_correction"),
        PluginCorrection(LogisticRegression()),
        IRLNL(LogisticRegression(), LogisticRegression()),
        LossCorrection(LogisticRegression()),
    ]
)
def test_all_noisy_estimators(estimator, check):
    return check(estimator)


class RandomSampleQuality(MetaEstimatorMixin, ClassifierMixin, BaseEstimator):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, **fit_params):
        check_classification_targets(y)

        _, untrusted = next(
            StratifiedShuffleSplit(train_size=0.5, random_state=1).split(X, y)
        )
        n_samples = _num_samples(X)
        sample_quality = np.ones(n_samples)
        sample_quality[untrusted] = 0
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y, sample_quality=sample_quality, **fit_params)

        if hasattr(self.estimator_, "n_features_in_"):
            self.n_features_in_ = self.estimator_.n_features_in_
        if hasattr(self.estimator_, "feature_names_in_"):
            self.feature_names_in_ = self.estimator_.feature_names_in_
        if hasattr(self.estimator_, "classes_"):
            self.classes_ = self.estimator_.classes_

        return self

    def decision_function(self, X):
        check_is_fitted(self)
        return self.estimator_.decision_function(X)

    def predict_proba(self, X):
        check_is_fitted(self)
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        check_is_fitted(self)
        return self.estimator_.predict(X)


@parametrize_with_checks(
    [
        IRBL(LogisticRegression(), LogisticRegression()),
        KPDR(LogisticRegression()),
        KKMM(LogisticRegression()),
        TrAdaBoostClassifier(LogisticRegression(), n_estimators=2),
        IPDR(LogisticRegression(), n_estimators=2),
        IKMM(LogisticRegression(), B=2, n_estimators=2),
        BiqualityBaseline(
            SelfTrainingClassifier(LogisticRegression(), max_iter=2),
            baseline="semi_supervised",
        ),
    ],
    expected_failed_checks=lambda estimator: {
        TrAdaBoostClassifier: SK_PER_ESTIMATOR_XFAIL_CHECKS.get(AdaBoostClassifier, {}),
        BiqualityBaseline: {
            "check_classifiers_classes": (
                "fails on a weird test case for semi_supervised baseline"
            )
        },
    }.get(type(estimator), {}),
)
def test_all_biquality_estimators(estimator, check):
    return check(RandomSampleQuality(estimator))
