"""Iterative Density Ratio Estimation for Biquality Learning."""

from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.special import xlogy
from sklearn.base import BaseEstimator, ClassifierMixin, clone, MetaEstimatorMixin
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.utils import check_array, check_scalar
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
        else hasattr(self.estimator_, attr)
    )


class IDR(BaseEstimator, ClassifierMixin, MetaEstimatorMixin, metaclass=ABCMeta):
    """Base class for Iterative Density Ratio Biquality Classifiers.

    A iterative density ratio classifier is a meta-algorithm that uses
    the covariate shift trick to reweight untrusted examples by using a
    density ratio estimation on the sample losses.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(
        self, estimator, n_estimators=10, exploit_iterative_learning=False, window=1
    ):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.exploit_iterative_learning = exploit_iterative_learning
        self.window = window

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

        X, y = self._validate_data(X, y, accept_sparse=True, force_all_finite=False)

        check_classification_targets(y)

        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        self.n_classes_ = len(self.classes_)
        y = self._le.transform(y)

        self._lb = LabelBinarizer().fit(y)
        Y = self._lb.transform(y)
        if Y.shape[1] == 1:
            Y = np.hstack((1 - Y, Y))

        n_samples = _num_samples(X)

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

        if self.exploit_iterative_learning and not hasattr(
            self.estimator, "warm_start"
        ):
            raise ValueError(
                "%s doesn't support iterative learning."
                % self.estimator.__class__.__name__
            )

        check_scalar(
            self.n_estimators, "n_estimators", int, min_val=1, include_boundaries="left"
        )
        check_scalar(self.window, "window", int, min_val=1, include_boundaries="left")

        estimator_param_names = self.estimator.get_params().keys()

        self.sample_weights_ = np.ones((n_samples, self.n_estimators))
        self._losses = np.empty((n_samples, self.n_estimators))
        self.estimator_ = clone(self.estimator)

        for i in range(0, self.n_estimators):
            if self.exploit_iterative_learning:
                if "n_estimators" in estimator_param_names:
                    self.estimator_.set_params(n_estimators=i + 1)
                if "max_iter" in estimator_param_names:
                    self.estimator_.set_params(max_iter=i + 1)
            else:
                self.estimator_ = clone(self.estimator)

            self.estimator_.fit(X, y, sample_weight=self.sample_weights_[:, i])

            if i < self.n_estimators - 1:
                y_pred = self.estimator_.predict_proba(X)
                eps = np.finfo(y_pred.dtype).eps
                np.clip(y_pred, eps, 1 - eps, out=y_pred)
                y_pred /= y_pred.sum(axis=1, keepdims=True)
                self._losses[:, i] = -xlogy(Y, y_pred).sum(axis=1)

                lw, rw = max(0, i + 1 - self.window), i + 1
                self.sample_weights_[sample_quality == 0, i + 1] = self._density_ratio(
                    self._losses[sample_quality == 0, lw:rw],
                    self._losses[sample_quality == 1, lw:rw],
                )

        return self

    @abstractmethod
    def _density_ratio(self, X_untrusted, X_trusted):
        """Implement density ratio estimation.

        Warning: This method needs to be overridden by subclasses.

        Parameters
        ----------
        X_untrusted : array-like, shape (n_samples_untrusted, window)
            The untrusted losses.

        X_trusted : array-like, shape (n_samples_trusted, window)
            The trusted losses.

        Returns
        -------
        density_ratio : array-like of shape (n_samples_untrusted,)
            The density ratios of the untrusted samples.
        """

    @available_if(_estimator_has("decision_function"))
    def decision_function(self, X):
        """Call decision function of the `final_estimator`.

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
