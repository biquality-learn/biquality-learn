"""Self-Paced Robust Learning for Leveraging Clean Labels in Noisy Data."""

from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.special import logsumexp, xlogy
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


class SelfPacedRobustLearning(BaseEstimator, ClassifierMixin, MetaEstimatorMixin):
    """Base class for Iterative Density Ratio Biquality Classifiers.

    A iterative density ratio classifier is a meta-algorithm that uses
    the covariate shift trick to reweight untrusted examples by using a
    density ratio estimation on the sample losses.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    def __init__(
        self,
        estimator,
        max_iter=10,
        tol=1e-4,
        init_threshold=0.1,
        learning_pace=1.1,
        max_threshold=3.5,
        exploit_iterative_learning=False,
    ):
        self.estimator = estimator
        self.init_threshold = init_threshold
        self.learning_pace = learning_pace
        self.max_threshold = max_threshold
        self.max_iter = max_iter
        self.tol = tol
        self.exploit_iterative_learning = exploit_iterative_learning

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

        estimator_param_names = self.estimator.get_params().keys()

        self.outers_ = []
        self.inners_ = []
        self.estimators_ = []

        for i in range(0, self.max_iter):
            if self.exploit_iterative_learning:
                if len(self.estimators_):
                    previous_estimator = self.estimators_[-1]

                    if "n_estimators" in estimator_param_names:
                        estimator = previous_estimator.set_params(n_estimators=i + 1)
                    if "max_iter" in estimator_param_names:
                        estimator = previous_estimator.set_params(max_iter=i + 1)

            else:
                estimator = clone(self.estimator)

            self.estimators_.append(estimator)
            self.inners_.append(self.inner(X, y, sample_quality))
            self.outers_.append(self.outer(X, y, sample_quality))

        return self

    def inner(self, X, y, sample_quality):
        n_samples = _num_samples(X)
        sample_weight = np.ones(n_samples)

        if len(self.outers_):
            sample_weight[sample_quality == 0] = 0
        else:
            y_pred = self.outers_[-1].decision_function(X)
            sample_weight[sample_quality == 0] = self.loss(y, y_pred) < self.threshold

        return sample_weight

    def outer(self, X, y, sample_quality):
        return self.estimators_[-1].fit(X, y, sample_weight=self.inners_[:, -1])

    def loss(self, y_true, y_pred):
        n_samples = y_true.shape[0]
        return logsumexp(y_pred, axis=1) - y_pred[np.arange(n_samples), y_true]

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
