"""Multiclass classification strategies"""

import warnings

import numpy as np
import scipy.sparse as sp
from joblib import delayed, Parallel
from sklearn.base import clone
from sklearn.multiclass import _ConstantPredictor, _estimators_has, OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_array, check_consistent_length
from sklearn.utils.metaestimators import available_if
from sklearn.utils.multiclass import _check_partial_fit_first_call
from sklearn.utils.validation import check_is_fitted, has_fit_parameter

__all__ = ["WeightedOneVsRestClassifier"]


def _fit_binary(estimator, X, y, sample_weight=None, classes=None):
    """Fit a single binary estimator."""
    unique_y = np.unique(y)
    if len(unique_y) == 1:
        if classes is not None:
            if y[0] == -1:
                c = 0
            else:
                c = y[0]
            warnings.warn(
                "Label %s is present in all training examples." % str(classes[c])
            )
        estimator = _ConstantPredictor().fit(X, unique_y)
    else:
        estimator = clone(estimator)
        if has_fit_parameter(estimator, "sample_weight"):
            estimator.fit(X, y, sample_weight=sample_weight)
        else:
            estimator.fit(X, y)
    return estimator


def _partial_fit_binary(estimator, X, y, sample_weight=None):
    """Partially fit a single binary estimator."""
    classes = np.array((0, 1))
    if has_fit_parameter(estimator, "sample_weight"):
        estimator.partial_fit(X, y, classes=classes, sample_weight=sample_weight)
    else:
        estimator.partial_fit(X, y, classes=classes)
    return estimator


class WeightedOneVsRestClassifier(OneVsRestClassifier):
    """A OneVsRestClassifier that supports reweighting samples per class.

    Parameters
    ----------
    estimator : estimator object
        An estimator object implementing :term:`fit` and one of
        :term:`decision_function` or :term:`predict_proba`.

    n_jobs : int, default=None
            The number of jobs to use for the computation: the `n_classes`
            one-vs-rest problems are computed in parallel.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors.

    verbose : int, default=0
        The verbosity level, if non zero, progress messages are printed.
        Below 50, the output is sent to stderr. Otherwise, the output is sent
        to stdout. The frequency of the messages increases with the verbosity
        level, reporting all iterations at 10. See :class:`joblib.Parallel` for
        more details.

    Attributes
    ----------
    estimators_ : list of `n_classes` estimators
        Estimators used for predictions.

    classes_ : array, shape = [`n_classes`]
        Class labels.

    n_classes_ : int
        Number of classes.

    label_binarizer_ : LabelBinarizer object
        Object used to transform multiclass labels to binary labels and
        vice-versa.

    multilabel_ : boolean
        Whether a OneVsRestClassifier is a multilabel classifier.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.
    """

    def __init__(self, estimator, *, n_jobs=None, verbose=0):
        super().__init__(estimator, n_jobs=n_jobs, verbose=verbose)

    def _validate_sample_weight(self, X, sample_weight):
        support_sample_weight = has_fit_parameter(self.estimator, "sample_weight")

        if sample_weight is not None:
            if not support_sample_weight:
                raise ValueError("The base estimator doesn't support sample weight")
            else:
                sample_weight = check_array(
                    sample_weight,
                    accept_sparse=False,
                    ensure_2d=False,
                    order="C",
                    input_name="sample_weight",
                )
                check_consistent_length(X, sample_weight)
                if sample_weight.ndim == 1:
                    return [sample_weight] * len(self.classes_)
                else:
                    return [col.ravel() for col in sample_weight.T]
        else:
            return [None] * len(self.classes_)

    def fit(self, X, y, sample_weight=None):
        """Fit underlying estimators.

        Parameters
        ----------
        X : (sparse) array-like of shape (n_samples, n_features)
            Data.

        y : (sparse) array-like of shape (n_samples,) or (n_samples, n_classes)
            Multi-class targets. An indicator matrix turns on multilabel
            classification.

        sample_weight : (sparse) array-like of shape (n_samples,) \
or (n_samples, n_classes), default=None
            Per class sample_weights. A matrix indicates that a different sample_weight
            is assigned to samples based on their target value.

        Returns
        -------
        self : object
            Instance of fitted estimator.
        """
        # A sparse LabelBinarizer, with sparse_output=True, has been shown to
        # outperform or match a dense label binarizer in all cases and has also
        # resulted in less or equal memory consumption in the fit_ovr function
        # overall.
        self.label_binarizer_ = LabelBinarizer()
        Y = self.label_binarizer_.fit_transform(y)
        if Y.shape[1] == 1:
            Y = np.hstack(((1 - Y), Y))
            self.label_binarizer_.y_type_ = "multiclass"
        Y = sp.csc_matrix(Y)
        self.classes_ = self.label_binarizer_.classes_
        columns = (col.toarray().ravel() for col in Y.T)

        sw_columns = self._validate_sample_weight(X, sample_weight)

        # In cases where individual estimators are very fast to train setting
        # n_jobs > 1 in can results in slower performance due to the overhead
        # of spawning threads.  See joblib issue #112.
        self.estimators_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_fit_binary)(
                self.estimator,
                X,
                column,
                sample_weight=sw_column,
                classes=[
                    "not %s" % self.label_binarizer_.classes_[i],
                    self.label_binarizer_.classes_[i],
                ],
            )
            for i, (sw_column, column) in enumerate(zip(sw_columns, columns))
        )

        if hasattr(self.estimators_[0], "n_features_in_"):
            self.n_features_in_ = self.estimators_[0].n_features_in_
        if hasattr(self.estimators_[0], "feature_names_in_"):
            self.feature_names_in_ = self.estimators_[0].feature_names_in_

        return self

    @available_if(_estimators_has("partial_fit"))
    def partial_fit(self, X, y, sample_weight=None, classes=None):
        """Partially fit underlying estimators.

        Should be used when memory is inefficient to train all data.
        Chunks of data can be passed in several iteration.

        Parameters
        ----------
        X : (sparse) array-like of shape (n_samples, n_features)
            Data.

        y : (sparse) array-like of shape (n_samples,) or (n_samples, n_classes)
            Multi-class targets. An indicator matrix turns on multilabel
            classification.

        sample_weight : (sparse) array-like of shape (n_samples,) \
or (n_samples, n_classes), default=None
            Per class sample_weights. A matrix indicates that a different sample_weight
            is assigned to samples based on their target value.

        classes : array, shape (n_classes, )
            Classes across all calls to partial_fit.
            Can be obtained via `np.unique(y_all)`, where y_all is the
            target vector of the entire dataset.
            This argument is only required in the first call of partial_fit
            and can be omitted in the subsequent calls.

        Returns
        -------
        self : object
            Instance of partially fitted estimator.
        """
        if _check_partial_fit_first_call(self, classes):
            if not hasattr(self.estimator, "partial_fit"):
                raise ValueError(
                    ("Base estimator {0}, doesn't have partial_fit method").format(
                        self.estimator
                    )
                )
            self.estimators_ = [clone(self.estimator) for _ in range(self.n_classes_)]

            # A sparse LabelBinarizer, with sparse_output=True, has been
            # shown to outperform or match a dense label binarizer in all
            # cases and has also resulted in less or equal memory consumption
            # in the fit_ovr function overall.
            self.label_binarizer_ = LabelBinarizer()
            self.label_binarizer_.fit(self.classes_)
            if self.label_binarizer_.y_type_ == "binary":
                self.label_binarizer_.y_type_ = "multiclass"

        if len(np.setdiff1d(y, self.classes_)):
            raise ValueError(
                (
                    "Mini-batch contains {0} while classes " + "must be subset of {1}"
                ).format(np.unique(y), self.classes_)
            )

        Y = self.label_binarizer_.transform(y)
        if Y.shape[1] == 1:
            Y = np.hstack(((1 - Y), Y))
        Y = sp.csc_matrix(Y)
        columns = (col.toarray().ravel() for col in Y.T)

        sw_columns = self._validate_sample_weight(X, sample_weight)

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_partial_fit_binary)(estimator, X, column, sw_column)
            for estimator, column, sw_column in zip(
                self.estimators_, columns, sw_columns
            )
        )

        if hasattr(self.estimators_[0], "n_features_in_"):
            self.n_features_in_ = self.estimators_[0].n_features_in_
        if hasattr(self.estimators_[0], "feature_names_in_"):
            self.feature_names_in_ = self.estimators_[0].feature_names_in_

        return self

    @available_if(_estimators_has("decision_function"))
    def decision_function(self, X):
        """Decision function for the OneVsRestClassifier.

        Return the distance of each sample from the decision boundary for each
        class. This can only be used with estimators which implement the
        `decision_function` method.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        T : array-like of shape (n_samples, n_classes) or (n_samples,) for \
            binary classification.
            Result of calling `decision_function` on the final estimator.

            .. versionchanged:: 0.19
                output shape changed to ``(n_samples,)`` to conform to
                scikit-learn conventions for binary classification.
        """
        check_is_fitted(self)
        scores = np.array(
            [est.decision_function(X).ravel() for est in self.estimators_]
        ).T
        if len(self.estimators_) == 2 and self.label_binarizer_.y_type_ == "multiclass":
            return (scores[:, 1] - scores[:, 0]) / 2
        else:
            return scores

    def _more_tags(self):
        return {
            "_xfail_checks": {
                **self.estimator._more_tags()["_xfail_checks"],
                "check_sample_weights_shape": "per class sample weights with ndim=2",
            }
        }
