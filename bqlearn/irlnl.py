"""Importance Reweighting for Learning with Label Noise."""

import numpy as np
from sklearn import clone
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin
from sklearn.preprocessing import LabelEncoder
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

__all__ = ["IRLNL"]


def _final_estimator_has(attr):
    """Check if we can delegate a method to the underlying estimator.
    First, we check the first fitted final estimator if available, otherwise we
    check the unfitted final estimator.
    """
    return lambda self: (
        hasattr(self.final_estimator_, attr)
        if hasattr(self, "final_estimator_")
        else hasattr(self.final_estimator, attr)
    )


class IRLNL(BaseEstimator, ClassifierMixin, MetaEstimatorMixin):
    """A Reweighted Classifier for Learning with Noisy Label [1]_.

    For each class :math:`y`, the untrusted samples are reweighted given the predictions
    made by a classifier learned on untrusted data and a noise transition matrix:

    .. math::

        \\frac{\mathbb{P}(Y=y|X)}{\mathbb{P}(\\tilde{Y}=y|X)} =
        \\frac{\mathbb{P}(\\tilde{Y}=y|X) - \mathbb{1}_{\\tilde{Y}=y} \\times \mathbb{P}
        (\\tilde{Y}= y | Y\\neq y ) - \mathbb{1}_{\\tilde{Y}\\neq y} \\times \mathbb{P}
        (\\tilde{Y}\\neq y | Y =y )}
        {\left(1 -  \mathbb{P}(\\tilde{Y}= y | Y\\neq y ) -
        \mathbb{P}(\\tilde{Y}\\neq y | Y =y )\\right)\mathbb{P}(\\tilde{Y}=y|X)}

    It does support multiclass classification thanks to a One versus Rest approach [2]_.

    Parameters
    ----------
    base_estimator : object, optional (default=None)
        The classifier used to estimate the transition matrix
        and the noisy classification task.
        Support for probability prediction is required.

    final_estimator : object, optional (default=None)
        The final estimator which will be reweighted to handle label noise.
        Support for sample weighting is required.

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
    final_estimator_ : classifier
        The final fitted estimator.

    transition_matrix_: ndarray of shape (n_classes, n_classes)
        Estimated transition matrix between untrusted and untrusted labels.

    sample_weight_ : ndarray, shape (n_samples, n_classes)
        The weights of the examples computed during :meth:`fit`.

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
    .. [1] T. Liu and D. Tao, "Classification with noisy labels \
        by importance reweighting.", \
        in IEEE Transactions on pattern analysis and machine intelligence, 2015

    .. [2] R. Wang, T. Liu and D. Tao, "Multiclass Learning With \
        Partially Corrupted Labels", \
        in IEEE Transactions on Neural Networks and Learning Systems, 2018.
    """

    def __init__(
        self,
        base_estimator,
        final_estimator,
        *,
        transition_matrix="anchor",
        quantile=0.97,
        n_iter=100,
        noise_free_prior=0,
        n_jobs=None,
    ):
        self.base_estimator = base_estimator
        self.final_estimator = final_estimator
        self.transition_matrix = transition_matrix
        self.quantile = quantile
        self.n_iter = n_iter
        self.noise_free_prior = noise_free_prior
        self.n_jobs = n_jobs

    def fit(self, X, y, sample_quality=None, **fit_params):
        """Fit the noisy classification model and the reweighted final classifier.

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

        if not hasattr(self.base_estimator, "predict_proba"):
            raise ValueError(
                """%s doesn't support predict_proba."""
                % self.estimator.__class__.__name__
            )

        if not has_fit_parameter(self.final_estimator, "sample_weight"):
            raise ValueError("The final estimator doesn't support sample weight")

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

        self.base_estimator_ = make_baseline(self.base_estimator, "untrusted_only")
        self.base_estimator_.fit(X, y, sample_quality=sample_quality, **fit_params)

        if isinstance(self.transition_matrix, str):
            self.transition_matrix_ = compute_transition_matrix(
                self.transition_matrix,
                self.base_estimator_,
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

        y_proba = self.base_estimator_.predict_proba(X)

        if self.n_classes_ == 2:
            rho_0 = self.transition_matrix_[1, 0]
            rho_1 = self.transition_matrix_[0, 1]

            y_proba = y * y_proba[:, 1] + (1 - y) * y_proba[:, 0]

            self.sample_weight_ = (y_proba - y * rho_1 - (1 - y) * rho_0) / (
                (1 - rho_0 - rho_1) * y_proba
            )

            self.final_estimator_ = clone(self.final_estimator)

        else:
            n_samples_untrusted = n_samples - np.count_nonzero(sample_quality)
            noisy_priors = (
                np.bincount(y[sample_quality == 0], minlength=self.n_classes_)
                / n_samples_untrusted
            )
            clean_priors = np.linalg.solve(self.transition_matrix_.T, noisy_priors)

            sample_weight = []

            for k in range(self.n_classes_):
                mask_k = y == k
                y_k = (mask_k).astype(int)
                y_proba_k = y_proba[:, k]
                y_proba_k[~mask_k] = 1 - y_proba_k[~mask_k]
                rho_0_k = 1 - self.transition_matrix_[k, k]
                rho_1_k = (
                    noisy_priors[k] - self.transition_matrix_[k, k] * clean_priors[k]
                ) / (1 - clean_priors[k])
                with np.errstate(divide="ignore", invalid="ignore"):
                    sample_weight_k = (
                        y_proba_k - y_k * rho_1_k - (1 - y_k) * rho_0_k
                    ) / ((1 - rho_0_k - rho_1_k) * y_proba_k)
                    sample_weight_k[y_proba_k == 0] = 0
                sample_weight.append(sample_weight_k)

            self.sample_weight_ = np.stack(sample_weight, axis=1)

            self.final_estimator_ = WeightedOneVsRestClassifier(
                self.final_estimator, n_jobs=self.n_jobs
            )

        self.sample_weight_[sample_quality == 1] = 1
        self.final_estimator_.fit(X, y, sample_weight=self.sample_weight_)

        return self

    @available_if(_final_estimator_has("decision_function"))
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
        return self.final_estimator_.decision_function(X)

    @available_if(_final_estimator_has("predict_proba"))
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
        return self.final_estimator_.predict_proba(X)

    @available_if(_final_estimator_has("predict"))
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
        return self._le.inverse_transform(self.final_estimator_.predict(X))

    def _more_tags(self):
        return {
            "_xfail_checks": {
                "check_parameters_default_constructible": "tofix",
            }
        }
