"""Classifiers with Symmetric loss functions
for learning under completly at random label noise"""

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model._base import LinearClassifierMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import column_or_1d
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import _check_sample_weight, check_is_fitted, check_scalar

__all__ = ["LinearUnhinged", "KernelUnhinged"]


class LinearUnhinged(BaseEstimator, LinearClassifierMixin):
    """Linear Unhinged Classification.

    Similar to KernelUnhinged with parameter kernel=’linear’, implemented
    using the primal formulation [1]_.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength; must be a positive float. Regularization
        only scales the weights. Larger values specify stronger regularization.
        Alpha corresponds to ``1 / (2C)`` in other linear models such as
        :class:`~sklearn.linear_model.LogisticRegression` or
        :class:`~sklearn.svm.LinearSVC`.

    Attributes
    ----------
    coef_ : ndarray of shape (1, n_features)
        Weights assigned to the features.

    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    n_classes_ : int
        The number of classes.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    References
    ----------
    .. [1] B. Rooyen, A. Menon and R. Williamson.\
           "Learning with Symmetric Label Noise: The Importance of Being Unhinged.",\
           NeurIPS, 2015
    """

    def __init__(self, *, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y, sample_weight=None):
        """Fit Linear Unhinged classification model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data. If kernel == "precomputed" this is instead
            a precomputed kernel matrix, of shape (n_samples, n_samples).

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        sample_weight : float or array-like of shape (n_samples,), default=None
            Individual weights for each sample, ignored if None is passed.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X, y = self._validate_data(
            X,
            y,
            accept_sparse=["csr"],
            order="C",
            accept_large_sparse=True,
        )

        check_classification_targets(y)

        check_scalar(
            self.alpha, "alpha", (float, int), min_val=0, include_boundaries="neither"
        )

        label_binarizer = LabelBinarizer(pos_label=1, neg_label=-1).fit(y)
        self.classes_ = label_binarizer.classes_
        self.n_classes_ = len(self.classes_)

        if self.n_classes_ > 2:
            raise ValueError("LinearUnhinged only supports binary classification.")

        y = label_binarizer.transform(y)
        y = column_or_1d(y)

        sample_weight = _check_sample_weight(sample_weight, X)

        sample_weight = sample_weight[..., None]
        y = y[..., None]

        if sp.issparse(X):
            self.coef_ = np.asarray(X.multiply(sample_weight * y).sum(axis=0))
        else:
            self.coef_ = (X * sample_weight * y).sum(axis=0)

        self.coef_ = 1 / (2 * self.alpha) * self.coef_.reshape(1, -1)
        self.intercept_ = 0.0

        return self

    def _more_tags(self):
        return {"binary_only": True}


class KernelUnhinged(BaseEstimator, ClassifierMixin):
    """Kernel Unhinged Classification.

    Kernel Unhinged Classification (KUC) [1]_ combines unhinged classification with the
    kernel trick. Fitting a KUC model can be done as class kernel mean maps.
    It's typically faster for medium sized datasets.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength; must be a positive float. Regularization
        only scales the weights. Larger values specify stronger regularization.
        Alpha corresponds to ``1 / (2C)`` in other linear models such as
        :class:`~sklearn.linear_model.LogisticRegression` or
        :class:`~sklearn.svm.LinearSVC`.

    kernel : str or callable, default="linear"
        Kernel mapping used internally. This parameter is directly passed to
        :class:`~sklearn.metrics.pairwise.pairwise_kernel`.
        If `kernel` is a string, it must be one of the metrics
        in `pairwise.PAIRWISE_KERNEL_FUNCTIONS`.
        If `kernel` is "precomputed", X is assumed to be a kernel matrix.
        Alternatively, if `kernel` is a callable function, it is called on
        each pair of instances (rows) and the resulting value recorded. The
        callable should take two rows from X as input and return the
        corresponding kernel value as a single number. This means that
        callables from :mod:`sklearn.metrics.pairwise` are not allowed, as
        they operate on matrices, not single samples. Use the string
        identifying the kernel instead.

    gamma : float, default=None
        Gamma parameter for the RBF, laplacian, polynomial, exponential chi2
        and sigmoid kernels. Interpretation of the default value is left to
        the kernel; see the documentation for sklearn.metrics.pairwise.
        Ignored by other kernels.

    degree : float, default=3
        Degree of the polynomial kernel. Ignored by other kernels.

    coef0 : float, default=1
        Zero coefficient for polynomial and sigmoid kernels.
        Ignored by other kernels.

    kernel_params : mapping of str to any, default=None
        Additional parameters (keyword arguments) for kernel function passed
        as callable object.

    Attributes
    ----------
    dual_coef_ : ndarray of shape (n_samples,) or (n_samples, n_targets)
        Representation of weight vector(s) in kernel space

    X_fit_ : {ndarray, sparse matrix} of shape (n_samples, n_features)
        Training data, which is also required for prediction. If
        kernel == "precomputed" this is instead the precomputed
        training matrix, of shape (n_samples, n_samples).

    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    n_classes_ : int
        The number of classes.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    References
    ----------
    .. [1] B. Rooyen, A. Menon and R. Williamson.\
           "Learning with Symmetric Label Noise: The Importance of Being Unhinged.",\
           NeurIPS, 2015
    """

    def __init__(
        self,
        *,
        alpha=1.0,
        kernel="linear",
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
    ):
        self.alpha = alpha
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params

    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma, "degree": self.degree, "coef0": self.coef0}
        return pairwise_kernels(X, Y, metric=self.kernel, filter_params=True, **params)

    def _more_tags(self):
        return {"pairwise": self.kernel == "precomputed", "binary_only": True}

    def fit(self, X, y, sample_weight=None):
        """Fit Kernel Unhinged classification model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data. If kernel == "precomputed" this is instead
            a precomputed kernel matrix, of shape (n_samples, n_samples).

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        sample_weight : float or array-like of shape (n_samples,), default=None
            Individual weights for each sample, ignored if None is passed.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Convert data
        X, y = self._validate_data(X, y, accept_sparse=("csr", "csc"))

        check_classification_targets(y)

        check_scalar(
            self.alpha, "alpha", (float, int), min_val=0, include_boundaries="neither"
        )

        label_binarizer = LabelBinarizer(pos_label=1, neg_label=-1).fit(y)
        self.classes_ = label_binarizer.classes_
        self.n_classes_ = len(self.classes_)

        if self.n_classes_ > 2:
            raise ValueError("KernelUnhinged only supports binary classification.")

        y = label_binarizer.transform(y)
        y = column_or_1d(y)

        sample_weight = _check_sample_weight(sample_weight, X)

        self.dual_coef_ = 1 / (2 * self.alpha) * y * sample_weight
        self.X_fit_ = X

        return self

    def decision_function(self, X):
        """Predict confidence scores for samples.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Samples. If kernel == "precomputed" this is instead a
            precomputed kernel matrix, shape = [n_samples,
            n_samples_fitted], where n_samples_fitted is the number of
            samples used in the fitting for this estimator.

        Returns
        -------
        C : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Returns predicted values.
        """
        check_is_fitted(self)
        X = self._validate_data(X, accept_sparse=("csr", "csc"), reset=False)
        K = self._get_kernel(X, self.X_fit_)
        return np.dot(K, self.dual_coef_)

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data matrix for which we want to get the predictions.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Vector containing the class labels for each sample.
        """
        scores = self.decision_function(X)
        indices = (scores > 0).astype(int)
        return self.classes_[indices]
