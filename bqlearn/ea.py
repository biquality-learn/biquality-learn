"""Frustratingly Easy approach to Domain Adaptation."""

import numpy as np
import scipy.sparse as sp
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, check_is_fitted, TransformerMixin
from sklearn.utils.validation import (
    _check_feature_names_in,
    _check_sample_weight,
    _num_samples,
    check_array,
)

__all__ = ["EasyADAPT"]


class EasyADAPT(BaseEstimator, TransformerMixin):
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

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`.

    References
    ----------
    .. [1] Daumé III, Hal. "Frustratingly Easy Domain Adaptation."\
        Proceedings of the 45th Annual Meeting of\
        the Association of Computational Linguistics. 2007.
    """

    def fit(self, X, y=None):
        """Fit the augmented model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The samples.

        y : None

        Returns
        -------
        self : object
            Returns self.
        """

        self._check_n_features(X, reset=True)
        self._check_feature_names(X, reset=True)

        return self

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "n_features_in_")
        feature_names_in = _check_feature_names_in(self, input_features)
        output_prefixes = ["common", "source", "target"]
        feature_names_out = []
        for output_prefix in output_prefixes:
            for feature_name_in in feature_names_in:
                feature_names_out.append(f"{output_prefix} {feature_name_in}")

        return feature_names_out

    def transform(self, X, sample_quality=None):
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

        X = check_array(X, accept_sparse=["csc", "csr", "lil"], force_all_finite=False)

        sample_quality = _check_sample_weight(sample_quality, X)

        n_samples = _num_samples(X)

        self._check_n_features(X, reset=False)
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

    def fit_transform(self, X, y=None, sample_quality=None):
        return self.fit(X, y).transform(X, sample_quality=sample_quality)

    def _more_tags(self):
        return {"no_validation": True}
