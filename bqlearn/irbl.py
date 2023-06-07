"""Importance Reweighting for Biquality Learning."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone, MetaEstimatorMixin
from sklearn.preprocessing import label_binarize
from sklearn.utils import check_array
from sklearn.utils.metaestimators import available_if
from sklearn.utils.multiclass import check_classification_targets, unique_labels
from sklearn.utils.validation import _num_samples, check_is_fitted, has_fit_parameter

from .baseline import make_baseline

__all__ = ["IRBL"]


def _final_estimator_has(attr):
    """Check if we can delegate a method to the underlying estimator.
    First, we check the fitted final estimator if available, otherwise we
    check the unfitted final estimator.
    """
    return lambda self: (
        hasattr(self.final_estimator_, attr)
        if hasattr(self, "final_estimator_")
        else hasattr(self.final_estimator, attr)
    )


class IRBL(BaseEstimator, ClassifierMixin, MetaEstimatorMixin):
    """A Reweighted Classifier for Biquality Learning.

    An IRBL [1]_ classifier is a is a meta-algorithm that uses the covariate shift
    trick to reweight untrusted examples from two classifiers
    learned on the trusted and untrusted dataset.

    Parameters
    ----------
    base_estimator : object, optional (default=None)
        The base estimator from which the IRBLClassifier is built.
        Support for probability prediction is required.

    final_estimator : object, optional (default=None)
        The final estimator from which the IRBLClassifier is built.
        Support for sample weighting is required.

    Attributes
    ----------
    final_estimator_ : classifier
        The final fitted estimator.

    sample_weight_ : ndarray, shape (n_samples,)
        The weights of the examples computed during :meth:`fit`.

    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    n_classes_ : int
        The number of classes.

    References
    ----------
    .. [1] P. Nodet, V. Lemaire, A. Bondu, A. Cornuéjols,\
        "Importance Reweighting for Biquality Learning", IJCNN, 2021.
    """

    def __init__(self, base_estimator, final_estimator):
        self.base_estimator = base_estimator
        self.final_estimator = final_estimator

    def fit(self, X, y, sample_quality=None):
        """Fit the reweighted model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).

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

        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)

        if sample_quality is not None:
            sample_quality = check_array(
                sample_quality, input_name="sample_quality", ensure_2d=False
            )
        else:
            raise ValueError("The 'sample_quality' parameter should not be None.")

        if not hasattr(self.base_estimator.__class__, "predict_proba"):
            raise ValueError(
                """%s doesn't support predict_proba."""
                % self.base_estimator.__class__.__name__
            )

        if not has_fit_parameter(self.final_estimator, "sample_weight"):
            raise ValueError(
                "%s doesn't support sample_weight."
                % self.final_estimator.__class__.__name__
            )

        self._estimator_trusted = make_baseline(self.base_estimator, "trusted_only")
        self._estimator_untrusted = make_baseline(self.base_estimator, "untrusted_only")

        self._estimator_trusted.fit(X, y, sample_quality=sample_quality)
        self._estimator_untrusted.fit(X, y, sample_quality=sample_quality)

        n_samples = _num_samples(X)
        Y = label_binarize(y, classes=self.classes_)
        if Y.shape[1] == 1:
            Y = np.hstack((1 - Y, Y))

        y_prob_trusted = self._estimator_trusted.predict_proba(X)
        y_prob_untrusted = self._estimator_untrusted.predict_proba(X)

        num = np.sum(y_prob_trusted * Y, axis=1)
        den = np.sum(y_prob_untrusted * Y, axis=1)

        self.sample_weight_ = np.divide(
            num,
            den,
            out=np.zeros(n_samples),
            where=den != 0,
        )
        self.sample_weight_[sample_quality == 1] = 1

        self.final_estimator_ = clone(self.final_estimator)
        self.final_estimator_.fit(X, y, sample_weight=self.sample_weight_)

        return self

    @available_if(_final_estimator_has("decision_function"))
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
        return self.final_estimator_.decision_function(X)

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
        return self.final_estimator_.predict(X)

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

    @available_if(_final_estimator_has("predict_log_proba"))
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
        return self.final_estimator_.predict_log_proba(X)
