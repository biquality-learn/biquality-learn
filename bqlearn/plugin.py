"""Plug-in correction to combat label noise."""

from copy import deepcopy

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import (
    _num_samples,
    check_array,
    check_is_fitted,
    check_scalar,
)

from bqlearn.baseline import make_baseline
from bqlearn.utils.validation import compute_transition_matrix

__all__ = ["PluginCorrection"]


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


class PluginCorrection(BaseEstimator, ClassifierMixin, MetaEstimatorMixin):
    """A Noise Corrected Plug-in Classifier.

    PluginCorrection [1]_ learns a classifier on noisy data
    and uses a noise transition matrix :math:`\mathbf{T}` at prediction time (plug-in)
    to correct the classifier.

    .. math:: \\forall x \in \mathcal{X}, f_T(x)=(\mathbf{T}^{t})^{-1}\cdot f_U(x)

    Parameters
    ----------
    estimator : estimator object
        An estimator object implementing :term:`fit` and :term:`predict_proba`.

    prefit : bool, default=False
        Whether a prefit model is expected to be passed into the constructor
        directly or not.
        If `True`, `estimator` must be a fitted estimator.
        If `False`, `estimator` is fitted by calling `fit`.

    transition_matrix : {'iterative', 'anchor', 'gold', 'confusion'} \
or array-like of shape (n_classes, n_classes), default='iterative'
        Algorithm to estimate the transition matrix.
        'gold' and 'confusion' are only available on biquality data.

    quantile : float, default=0.97
        Quantile used to select the anchor points.
        Only used when `transition_matrix='anchor'` or `transition_matrix='iterative'`.

    n_iter : int, default=100
        Number of iteratives to compute the transition matrix.
        Only used when `transition_matrix='iterative'`.

    noise_free_prior : float, default=0.0
        Factor for the convex combination between the estimated transition_matrix
        and the identity matrix to lower the condition number of the estimated
        transition matrix. It's equivalent to take a more conservative noise-free
        prior.

    random_state : int or RandomState, default=None
        Controls the random seed given at base_estimator.
        Pass an int for reproducible output across multiple function calls.

    Attributes
    ----------
    estimator_ : classifier
        The fitted estimator.

    transition_matrix_: ndarray of shape (n_classes, n_classes)
        Estimated transition matrix between untrusted and untrusted labels.

    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    n_classes_ : int
        The number of classes.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

    References
    ----------
    .. [1] M. Zhang, J. Lee, and S. Agarwal.\
        "Learning from noisy labels with no change to the training process.",\
        ICML, 2021.
    """

    def __init__(
        self,
        estimator,
        *,
        prefit=False,
        transition_matrix="iterative",
        quantile=0.97,
        n_iter=100,
        noise_free_prior=0.0,
    ):
        self.estimator = estimator
        self.prefit = prefit
        self.transition_matrix = transition_matrix
        self.quantile = quantile
        self.n_iter = n_iter
        self.noise_free_prior = noise_free_prior

    @available_if(_estimator_has("predict_proba"))
    def decision_function(self, X):
        """Noise-corrected plug-in of predicted probabilites from `estimator`.

        For binary classification, the score is the margin between the two classes.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples, n_classes)
            The predicted classes.
        """
        check_is_fitted(self)
        scores = self.estimator_.predict_proba(X) @ self._inv_transition_matrix
        if scores.shape[1] == 2:
            return scores[:, 1] - scores[:, 0]
        else:
            return scores

    @available_if(_estimator_has("predict_proba"))
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
        scores = self.decision_function(X)
        if scores.ndim == 1:
            indices = (scores > 0).astype(int)
        else:
            indices = scores.argmax(axis=1)
        return self.classes_[indices]

    def fit(self, X, y, sample_quality=None):
        """Fit the noise corrected plug-in classification model.

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

        check_scalar(self.quantile, "quantile", (float, int), min_val=0, max_val=1)
        check_scalar(
            self.noise_free_prior,
            "noise_free_prior",
            (float, int),
            min_val=0,
            max_val=1,
        )

        X, y = self._validate_data(
            X, y, accept_sparse=["csr", "csc", "lil", "dok"], force_all_finite=False
        )

        n_samples = _num_samples(X)

        if sample_quality is None:
            sample_quality = np.zeros(n_samples, dtype=np.float32)

        sample_quality = check_array(
            sample_quality, input_name="sample_quality", ensure_2d=False
        )

        if (
            isinstance(self.transition_matrix, str)
            and self.transition_matrix
            in [
                "gold",
                "confusion",
            ]
            and np.all(sample_quality == 0)
        ):
            raise ValueError(
                f"Unsupported transition matrix : {self.transition_matrix}"
                " without high quality samples."
            )

        if not hasattr(self.estimator, "predict_proba"):
            raise ValueError(
                """%s doesn't support predict_proba."""
                % self.estimator.__class__.__name__
            )

        if self.prefit:
            try:
                check_is_fitted(self.estimator)
            except NotFittedError as exc:
                raise NotFittedError(
                    "When `prefit=True`, `estimator` is expected to be a fitted "
                    "estimator."
                ) from exc
            self.estimator_ = deepcopy(self.estimator)
        else:
            self.estimator_ = make_baseline(self.estimator, "untrusted_only")
            self.estimator_.fit(X, y, sample_quality)

        if isinstance(self.transition_matrix, str):
            self.transition_matrix_ = compute_transition_matrix(
                self.transition_matrix,
                self.estimator_,
                X,
                y,
                sample_quality,
                quantile=self.quantile,
                n_iter=self.n_iter,
            )
        else:
            self.transition_matrix_ = check_array(self.transition_matrix, copy=True)

        self.classes_ = self.estimator_.classes_
        self.n_classes_ = len(self.classes_)

        self.transition_matrix_ -= self.noise_free_prior * (
            self.transition_matrix_ - np.identity(self.n_classes_)
        )

        self._inv_transition_matrix = np.linalg.inv(self.transition_matrix_)

        if hasattr(self.estimator_, "n_features_in_"):
            self.n_features_in_ = self.estimator_.n_features_in_
        if hasattr(self.estimator_, "feature_names_in_"):
            self.feature_names_in_ = self.estimator_.feature_names_in_

        return self
