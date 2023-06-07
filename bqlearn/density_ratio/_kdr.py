"""K-Density Ratio Estimation for Biquality Learning."""

from abc import ABCMeta, abstractmethod

import numpy as np
from joblib import delayed, Parallel
from sklearn.base import BaseEstimator, ClassifierMixin, clone, MetaEstimatorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_array, safe_mask
from sklearn.utils.metaestimators import available_if
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import _num_samples, check_is_fitted, has_fit_parameter


def _estimator_has(attr):
    """Check if we can delegate a method to the underlying estimator.
    First, we check the first fitted final estimator if available, otherwise we
    check the unfitted final estimator.
    """
    return lambda self: (
        hasattr(self.estimator_, attr)
        if hasattr(self, "estimator_")
        else hasattr(self.estimator, attr)
    )


class KDR(BaseEstimator, ClassifierMixin, MetaEstimatorMixin, metaclass=ABCMeta):
    """Base class for K Density Ratio Biquality Classifiers.

    A K density ratio classifier is a meta-algorithm that uses the covariate shift
    trick to reweight untrusted examples by using a ratio density estimation done on
    the features conditionally to the labels.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(self, estimator=None, n_jobs=None):
        self.estimator = estimator
        self.n_jobs = n_jobs

    def fit(self, X, y, sample_quality=None):
        """Fit the reweighted model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        y : array-like of shape (n_samples,)
            The target labels.

        sample_quality : array-like, shape (n_samples,)
            Sample qualities.

        Returns
        -------
        self : object
        """

        X, y = self._validate_data(
            X, y, accept_sparse=["csr", "csc", "lil", "dok"], force_all_finite=False
        )

        check_classification_targets(y)

        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        self.n_classes_ = len(self.classes_)
        y = self._le.transform(y)

        if sample_quality is not None:
            sample_quality = check_array(
                sample_quality, input_name="sample_quality", ensure_2d=False
            )
        else:
            raise ValueError("The 'sample_quality' parameter should not be None.")

        if not has_fit_parameter(self.estimator, "sample_weight"):
            raise ValueError(
                "%s doesn't support sample_weight." % self.estimator.__class__.__name__
            )

        # Division by zero means no untrusted samples so no weights will be computed
        with np.errstate(divide="ignore", invalid="ignore"):
            priors = (
                np.bincount(y[sample_quality == 1], minlength=self.n_classes_)
                / np.count_nonzero(sample_quality == 1)
            ) / (
                np.bincount(y[sample_quality == 0], minlength=self.n_classes_)
                / np.count_nonzero(sample_quality == 0)
            )

        def compute_density_ratios(self, X, y, sample_quality, i):
            X_trusted_i = X[safe_mask(X, (y == i) * (sample_quality == 1))]
            if _num_samples(X_trusted_i) == 0:
                raise ValueError(f"No trusted samples for class {self.classes_[i]}")
            X_untrusted_i = X[safe_mask(X, (y == i) * (sample_quality == 0))]
            if _num_samples(X_untrusted_i) > 0:
                return self._density_ratio(X_untrusted_i, X_trusted_i)
            else:
                return np.array([])

        density_ratios = Parallel(n_jobs=self.n_jobs)(
            delayed(compute_density_ratios)(self, X, y, sample_quality, i)
            for i in range(self.n_classes_)
        )

        n_samples = _num_samples(X)
        self.sample_weight_ = np.ones(n_samples)

        for i in range(self.n_classes_):
            self.sample_weight_[(y == i) * (sample_quality == 0)] = (
                density_ratios[i] * priors[i]
            )

        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y, sample_weight=self.sample_weight_)

        return self

    @abstractmethod
    def _density_ratio(self, X_untrusted_i, X_trusted_i):
        """Implement density ratio estimation.

        Warning: This method needs to be overridden by subclasses.

        Parameters
        ----------
        X_untrusted_i : array-like, shape (n_samples_untrusted_i, n_features)
            The untrusted samples of class i.

        X_trusted_i : array-like, shape (n_samples_trusted_i, n_features)
            The trusted samples of class i.

        Returns
        -------
        density_ratio : array-like of shape (n_samples_untrusted_i,)
            The density ratios of the untrusted samples of class i.
        """

    @available_if(_estimator_has("decision_function"))
    def decision_function(self, X):
        """Call decision function of the `estimator`.

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
        return self.estimator_.decision_function(X)

    @available_if(_estimator_has("predict"))
    def predict(self, X):
        """Predict the classes of `X`.

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
        return self._le.inverse_transform(self.estimator_.predict(X))

    @available_if(_estimator_has("predict_proba"))
    def predict_proba(self, X):
        """Predict probability for each possible outcome.

        Parameters
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
        return self.estimator_.predict_proba(X)

    @available_if(_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X):
        """Predict log probability for each possible outcome.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        log_p : array, shape (n_samples, n_classes)
            Array with log prediction probabilities.
        """
        check_is_fitted(self)
        return self.estimator_.predict_log_proba(X)
