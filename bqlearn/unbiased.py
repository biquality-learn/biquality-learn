"""Method of Unbiased Estimators."""


import numpy as np
import scipy.sparse as sp
from scipy.sparse import issparse
from sklearn import clone
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin
from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.utils import check_scalar
from sklearn.utils.metaestimators import available_if
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import (
    _num_samples,
    check_array,
    check_is_fitted,
    has_fit_parameter,
)

from bqlearn.baseline import make_baseline
from bqlearn.multiclass import WeightedOneVsRestClassifier
from bqlearn.utils.validation import compute_transition_matrix

__all__ = ["LossCorrection"]


def _estimator_has(attr):
    """Check if we can delegate a method to the underlying estimator.
    First, we check the first fitted estimator if available, otherwise we
    check the unfitted estimator.
    """
    return lambda self: (
        hasattr(self.estimator_, attr)
        if hasattr(self, "estimator_")
        else hasattr(self.estimator, attr)
    )


class LossCorrection(BaseEstimator, ClassifierMixin, MetaEstimatorMixin):
    """A Classifier corrected with the method of unbiased estimators [1]_.

    It construts a surrogate loss :math:`\\tilde{L}` from the loss of interest :math:`L`
    such that :math:`\mathbb{E}_{\\tilde{y}}[\\tilde{L}(f(x),\\tilde{y})] = L(f(x),y)`.

    .. math::

        \\tilde{L}(f(x),y) = \\frac{(1-\mathbb{P}(\\tilde{Y}= y|Y\\neq ))L(f(x), y)
        - \mathbb{P}(\\tilde{Y}\\neq y | Y =y ) L(f(x), -y) }
        {1 - \mathbb{P}(\\tilde{Y}= y| Y\\neq y ) - \mathbb{P}(\\tilde{Y}\\neq |Y =y)}

    It does support multiclass classification thanks to a One versus Rest approach.

    Parameters
    ----------
    estimator : object, optional (default=None)
        The estimator which will be corrected to handle label noise.
        Support for negative sample weighting is required.
        Support for probability prediction for certain methods of transition matrix
        estimation.

    transition_matrix : {'iterative', 'anchor', 'gold', 'confusion'} \
or array-like of shape (n_classes, n_classes), default='anchor'
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

    n_jobs : int, default=None
            The number of jobs to use for the computation: the `n_classes`
            one-vs-rest problems are computed in parallel.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors.

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
    .. [1] N. Natarajan, I. S. Dhillon, P. Ravikumar, and A. Tewari,\
        "Learning with Noisy Labels", NeurIPS, 2013.
    """

    def __init__(
        self,
        estimator,
        *,
        transition_matrix="anchor",
        quantile=0.97,
        n_iter=100,
        noise_free_prior=0,
        n_jobs=None,
    ):
        self.estimator = estimator
        self.transition_matrix = transition_matrix
        self.quantile = quantile
        self.n_iter = n_iter
        self.noise_free_prior = noise_free_prior
        self.n_jobs = n_jobs

    def fit(self, X, y, sample_quality=None, **fit_params):
        """Fit the noisy transition matrix and the corrected classifier.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : object
            Returns the instance itself.
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

        check_classification_targets(y)

        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        self.n_classes_ = len(self.classes_)
        y = self._le.transform(y)

        n_samples = _num_samples(X)

        if sample_quality is None:
            sample_quality = np.zeros(n_samples, dtype=np.float32)

        sample_quality = check_array(
            sample_quality, input_name="sample_quality", ensure_2d=False
        )

        if not has_fit_parameter(self.estimator, "sample_weight"):
            raise ValueError("The estimator doesn't support sample weight")

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

        if isinstance(self.transition_matrix, str):
            self._noisy_estimator = make_baseline(self.estimator, "untrusted_only")
            self._noisy_estimator.fit(X, y, sample_quality=sample_quality, **fit_params)
            self.transition_matrix_ = compute_transition_matrix(
                self.transition_matrix,
                self._noisy_estimator,
                X,
                y,
                sample_quality,
                quantile=self.quantile,
                n_iter=self.n_iter,
            )
        else:
            self.transition_matrix_ = check_array(self.transition_matrix, copy=True)

        self.transition_matrix_ -= self.noise_free_prior * (
            self.transition_matrix_ - np.identity(self.n_classes_)
        )

        if self.n_classes_ == 2:
            rho_0 = self.transition_matrix_[1, 0]
            rho_1 = self.transition_matrix_[0, 1]

            sample_weight = (1 - rho_1) * y + (1 - rho_0) * (1 - y)
            sample_weight /= 1 - rho_1 - rho_0

            negative_sample_weight = -rho_0 * y - rho_1 * (1 - y)
            negative_sample_weight /= 1 - rho_1 - rho_0

            self.estimator_ = clone(self.estimator)

        else:
            n_samples_untrusted = n_samples - np.count_nonzero(sample_quality)

            noisy_priors = (
                np.bincount(y[sample_quality == 0], minlength=self.n_classes_)
                / n_samples_untrusted
            )
            clean_priors = np.linalg.solve(self.transition_matrix_.T, noisy_priors)

            sample_weight = []
            negative_sample_weight = []

            for k in range(self.n_classes_):
                mask_k = y == k
                y_k = (mask_k).astype(int)

                rho_0_k = 1 - self.transition_matrix_[k, k]
                rho_1_k = (
                    noisy_priors[k] - self.transition_matrix_[k, k] * clean_priors[k]
                ) / (1 - clean_priors[k])

                sample_weight_k = (1 - rho_1_k) * y_k + (1 - rho_0_k) * (1 - y_k)
                sample_weight_k /= 1 - rho_1_k - rho_0_k

                negative_sample_weight_k = -rho_0_k * y_k - rho_1_k * (1 - y_k)
                negative_sample_weight_k /= 1 - rho_1_k - rho_0_k

                sample_weight.append(sample_weight_k)
                negative_sample_weight.append(negative_sample_weight_k)

            sample_weight = np.stack(sample_weight, axis=1)
            negative_sample_weight = np.stack(negative_sample_weight, axis=1)

            self.estimator_ = WeightedOneVsRestClassifier(
                self.estimator, n_jobs=self.n_jobs
            )

        sample_weight[sample_quality == 1] = 1
        negative_sample_weight[sample_quality == 1] = 0

        Y = label_binarize(y, classes=range(self.n_classes_))
        Y = np.vstack((Y, 1 - Y))
        if self.n_classes_ == 2:
            Y = Y.ravel()

        if issparse(X):
            X = sp.vstack((X, X))
        else:
            X = np.vstack((X, X))

        self.sample_weight_ = np.concatenate(
            (sample_weight, negative_sample_weight), axis=0
        )

        self.estimator_.fit(X, Y, sample_weight=self.sample_weight_)

        return self

    @available_if(_estimator_has("decision_function"))
    def decision_function(self, X):
        """Call predict of the regressor `estimator`.

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
        return self.estimator_.decision_function(X)

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
        scores = self.estimator_.predict_proba(X)
        scores /= scores.sum(axis=1, keepdims=True)
        return scores

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
        try:
            scores = self.estimator_.decision_function(X)
            if scores.ndim == 1:
                scores = scores.reshape(-1, 1)
                scores = np.hstack([-scores, scores])
        except (AttributeError, NotImplementedError):
            scores = self.estimator_.predict_proba(X)
        return self._le.inverse_transform(scores.argmax(axis=1))
