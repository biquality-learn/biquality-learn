"""Frustratingly Easy approach to Domain Adaptation."""

import numpy as np
import scipy.sparse as sp
from scipy.sparse import issparse
from sklearn.base import (
    BaseEstimator,
    check_is_fitted,
    ClassifierMixin,
    clone,
    MetaEstimatorMixin,
)
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import _check_sample_weight, _num_samples, check_array

__all__ = ["EasyADAPT"]


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


# TODO: Refactor into a augmenter when metadata routing is done for pipelines
# This augmenter will be stateless (no fit) and augment X accordingly to the routed
# sample_quality values. We will get the same behaviour as the current estimator by
# pipelining the new augmenter and an estimator.
# https://github.com/scikit-learn/scikit-learn/pull/24270
class EasyADAPT(BaseEstimator, ClassifierMixin, MetaEstimatorMixin):
    """A Frustratingly Easy approach to Domain Adaptation.

    EasyADAPT [1]_ creates an augmented input space
    :math:`\\tilde{\mathcal{X}} = \mathcal{X}^3`
    with two different mapping for untrusted and trusted samples,
    :math:`\Psi_U:\mathcal{X}\mapsto \\tilde{\mathcal{X}}` and
    :math:`\Psi_T:\mathcal{X}\mapsto \\tilde{\mathcal{X}}`.

    -  ..math::
        \\forall \mathbf{x} \in \mathcal{X},
        \Psi_U(\mathbf{x})=<\mathbf{x}, \mathbf{x}, \mathbf{0}>

    -  ..math::
        \\forall \mathbf{x} \in \mathcal{X},
        \Psi_T(\mathbf{x})=<\mathbf{x}, \mathbf{0}, \mathbf{x}>

    This augmented domain :math:`\\tilde{\mathcal{X}}` allow for the classifier to learn
    different relation between the features and the target differently
    for the untrusted, trusted and general domain.

    Parameters
    ----------
    estimator : estimator object
        An estimator object.

    Attributes
    ----------
    estimator_ : classifier
        The fitted estimator.

    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    n_classes_ : int
        The number of classes. Only defined if the
        underlying estimator exposes such an attribute when fit.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

    References
    ----------
    .. [1] Daumé III, Hal. "Frustratingly Easy Domain Adaptation."\
        Proceedings of the 45th Annual Meeting of\
        the Association of Computational Linguistics. 2007.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, sample_quality=None):
        """Fit the augmented model.

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

        X = self._validate_data(X, accept_sparse=["csr"], force_all_finite=False)

        X_aug = self.augment(X, sample_quality=sample_quality)

        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X_aug, y)

        self.classes_ = self.estimator_.classes_

        return self

    def augment(self, X, sample_quality=None):
        """Augment the input dataset according to `sample_quality`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The samples.

        sample_quality : array-like, shape (n_samples,)
            Per-sample qualities.

        Returns
        -------
        X_aug : array-like of shape (n_samples, 3 * n_features)
            Returns the augmented samples.
        """

        X = check_array(X, accept_sparse=["csr"], force_all_finite=False)

        sample_quality = _check_sample_weight(sample_quality, X)

        n_samples = _num_samples(X)
        n_features = self.n_features_in_

        if issparse(X):
            X_aug = sp.csr_matrix((n_samples, 3 * n_features))
        else:
            X_aug = np.zeros((n_samples, 3 * n_features))

        X_aug[:, 0:n_features] = X
        X_aug[sample_quality == 0, n_features : 2 * n_features] = X[
            sample_quality == 0, :
        ]
        X_aug[sample_quality == 1, 2 * n_features : 3 * n_features] = X[
            sample_quality == 1, :
        ]
        return X_aug

    @available_if(_estimator_has("decision_function"))
    def decision_function(self, X):
        """Call decision function of the `estimator` on the augmented dataset.

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
        X_aug = self.augment(X)

        return self.estimator_.decision_function(X_aug)

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
        X_aug = self.augment(X)

        return self.estimator_.predict(X_aug)

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
        X_aug = self.augment(X)

        return self.estimator_.predict_proba(X_aug)

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
        X_aug = self.augment(X)

        return self.estimator_.predict_log_proba(X_aug)
