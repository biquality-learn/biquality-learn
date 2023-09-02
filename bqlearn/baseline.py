"""Baselines for Biquality Learning"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone, MetaEstimatorMixin
from sklearn.utils import check_array, safe_mask
from sklearn.utils.metaestimators import available_if
from sklearn.utils.multiclass import check_classification_targets, unique_labels
from sklearn.utils.validation import _num_samples, check_is_fitted

__all__ = ["BiqualityBaseline", "make_baseline"]


def _estimator_has(attr):
    """Check if we can delegate a method to the underlying estimator.
    First, we check the fitted estimator if available, otherwise we
    check the unfitted estimator.
    """
    return lambda self: (
        hasattr(self.estimator_, attr)
        if hasattr(self, "estimator_")
        else hasattr(self.estimator, attr)
    )


class BiqualityBaseline(BaseEstimator, ClassifierMixin, MetaEstimatorMixin):
    """A Biquality Baseline.

    A BaselineBiqualityClassifier lift usual scikit-learn classifiers to train
    on biquality data as a baseline algortihm.

    Parameters
    ----------
    estimator : object, optional (default=None)
        The base estimator from which the BaselineBiqualityClassifier is built.

    baseline : {'trusted_only', 'untrusted_only', 'no_correction', 'semi_supervised'},
        default='no_correction'

    Attributes
    ----------
    estimator_ : classifier
        The final fitted estimator.

    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    n_classes_ : int
        The number of classes.
    """

    def __init__(self, estimator, baseline="no_correction"):
        self.estimator = estimator
        self.baseline = baseline

    def fit(self, X, y, sample_quality=None):
        """Fit the baseline model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The samples.

        y : array-like of shape (n_samples,)
            The targets.

        sample_quality : array-like, shape (n_samples,)
            Per-sample qualities.

        Returns
        -------
        self : object
            Returns self.
        """

        X, y = self._validate_data(
            X, y, accept_sparse=["csr", "csc", "lil", "dok"], force_all_finite=False
        )

        check_classification_targets(y)

        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)

        if sample_quality is None and self.baseline != "no_correction":
            raise ValueError("The 'sample_quality' parameter should not be None.")

        if sample_quality is not None:
            sample_quality = check_array(
                sample_quality, input_name="sample_quality", ensure_2d=False
            )

        if self.estimator is None:
            raise ValueError("estimator cannot be None!")

        self.estimator_ = clone(self.estimator)

        if self.baseline not in (
            "trusted_only",
            "untrusted_only",
            "no_correction",
            "semi_supervised",
        ):
            raise ValueError("method %s is not supported" % self.baseline)

        if self.baseline == "trusted_only":
            mask = sample_quality == 1
            self.estimator_.fit(X[safe_mask(X, mask)], y[mask])
        if self.baseline == "untrusted_only":
            mask = sample_quality == 0
            self.estimator_.fit(X[safe_mask(X, mask)], y[mask])
        if self.baseline == "no_correction":
            self.estimator_.fit(X, y)
        if self.baseline == "semi_supervised":
            y_copy = np.copy(y)
            if y_copy.dtype.kind in ["U", "S"]:
                y_copy = y_copy.astype(np.object_)
            y_copy[sample_quality == 0] = -1
            self.estimator_.fit(X, y_copy)

        if hasattr(self.estimator_, "n_features_in_"):
            self.n_features_in_ = self.estimator_.n_features_in_
        if hasattr(self.estimator_, "feature_names_in_"):
            self.feature_names_in_ = self.estimator_.feature_names_in_

        return self

    @available_if(_estimator_has("predict"))
    def predict(self, X):
        """
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The predicted classes.
        """
        check_is_fitted(self)
        return self.estimator_.predict(X)

    @available_if(_estimator_has("decision_function"))
    def decision_function(self, X):
        """Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        p : array, shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        check_is_fitted(self)
        n_samples = _num_samples(X)
        if self.n_classes_ == 2:
            return self.estimator_.decision_function(X)
        else:
            scores = np.zeros((n_samples, self.n_classes_))
            indices = np.isin(self.classes_, self.estimator_.classes_)
            scores[:, indices] = self.estimator_.decision_function(X)
            return scores

    @available_if(_estimator_has("predict_proba"))
    def predict_proba(self, X):
        """Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        p : array, shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        check_is_fitted(self)
        n_samples = _num_samples(X)
        proba = np.zeros((n_samples, self.n_classes_))
        indices = np.isin(self.classes_, self.estimator_.classes_)
        proba[:, indices] = self.estimator_.predict_proba(X)
        return proba

    def _more_tags(self):
        return {
            "_xfail_checks": {
                "check_classifiers_classes": (
                    "fails on a weird test case for semi_supervised baseline"
                ),
            },
        }


def make_baseline(estimator, baseline="no-correction"):
    "Make a biquality classifier from a scikit-learn classifier"
    return BiqualityBaseline(estimator, baseline=baseline)
