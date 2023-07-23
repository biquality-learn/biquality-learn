"""Tradaboost, a modified Adaboost algorithm for Transfer Learning"""

import math
import numbers
import warnings

import numpy as np
from sklearn.base import is_regressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble._weight_boosting import BaseWeightBoosting
from sklearn.svm import LinearSVC
from sklearn.utils import check_random_state, check_scalar
from sklearn.utils.validation import (
    _check_sample_weight,
    check_array,
    has_fit_parameter,
)

__all__ = ["TrAdaBoostClassifier"]


class TrAdaBoostClassifier(AdaBoostClassifier):
    """A TrAdaBoost classifier.

    A  TrAdaBoost [1]_ classifier is a meta-estimator that adapts Adaboost [2]_ to
    transfert learning. For the trusted dataset, TrAdaBoost works exactly the same way
    as AdaBoost, the misclassified examples get a higher weight for estimators
    to focus on them. For the untrusted dataset, TrAdaBoost works the opposite way
    as in WMA [3]_, the misclassified examples are deemed useless for the task
    and thus see their weights decreased.

    This class implements a modified TrAdaBoost for multi-class classification
    in the fashion of AdaBoost-SAMME [4]_ with weight drift correction from
    Dynamic TrAdaBoost [5]_.

    Parameters
    ----------
    estimator : object, default=None
        The base estimator from which the reversed boosted ensemble is built.
        Support for sample weighting is required. If ``None``, then
        the base estimator is :class:`~sklearn.linear_model.LinearSVC`.

    n_estimators : int
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.

    learning_rate : float, default=1.0
        Learning rate shrinks the contribution of each classifier by
        ``learning_rate``. There is a trade-off between ``learning_rate`` and
        ``n_estimators``

    random_state : int or RandomState, default=None
        Controls the random seed given at each estimator at each
        boosting iteration. Pass an int for reproducible output across
        multiple function calls.

    Attributes
    ----------
    estimator_ : estimator
        The base estimator from which the ensemble is grown.

    estimators_ : list of classifiers
        The collection of fitted sub-estimators.

    n_features_in_ : int
        The number of features seen during :meth:`fit`.

    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    n_classes_ : int
        The number of classes.

    estimator_weights_ : ndarray of floats
        Weights for each estimator in the boosted ensemble.

    estimator_errors_ : ndarray of floats
        Classification error for each estimator in the boosted
        ensemble.

    References
    ----------
    .. [1] Wenyuan Dai, Qiang Yang, Gui-Rong Xue, Yong Yu,\
        "Boosting for Transfer Learning", 2007.

    .. [2] Y. Freund, R. Schapire, "A Decision-Theoretic Generalization of\
        on-Line Learning and an Application to Boosting", 1995.

    .. [3] N. Littlestone, M.K. Warmuth, "The Weighted Majority Algorithm", 1994.

    .. [4] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost", 2009.

    .. [5] S. Al-Stouhi and C. K. Reddy, "Adaptive boosting for transfer learning\
        using dynamic updates", ECML, 2011.
    """

    def __init__(
        self,
        estimator=None,
        *,
        n_estimators=50,
        learning_rate=1.0,
        random_state=None,
    ):
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            algorithm="SAMME",
            random_state=random_state,
        )

    # TODO: delete code when https://github.com/scikit-learn/scikit-learn/pull/24026
    # is merged
    def fit(self, X, y, sample_weight=None, sample_quality=None, **fit_params):
        """Build a boosted classifier/regressor from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.

        sample_quality : array-like, shape (n_samples,)
            Sample qualities.

        Returns
        -------
        self : object
        """
        # Validate scalar parameters
        check_scalar(
            self.n_estimators,
            "n_estimators",
            target_type=numbers.Integral,
            min_val=1,
            include_boundaries="left",
        )

        check_scalar(
            self.learning_rate,
            "learning_rate",
            target_type=numbers.Real,
            min_val=0,
            include_boundaries="neither",
        )

        X, y = self._validate_data(
            X,
            y,
            accept_sparse=["csr", "csc"],
            ensure_2d=True,
            allow_nd=True,
            dtype=None,
            y_numeric=is_regressor(self),
        )

        sample_weight = _check_sample_weight(
            sample_weight, X, np.float64, copy=True, only_non_negative=True
        )
        sample_weight /= sample_weight.sum()

        if sample_quality is not None:
            sample_quality = check_array(
                sample_quality, input_name="sample_quality", ensure_2d=False
            )
        else:
            raise ValueError("The 'sample_quality' parameter should not be None.")

        # Check parameters
        self._validate_estimator()

        # Clear any previous fit results
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)

        # Initialization of the random number instance that will be used to
        # generate a seed at each iteration
        random_state = check_random_state(self.random_state)

        for iboost in range(self.n_estimators):
            # Boosting step
            sample_weight, estimator_weight, estimator_error = self._boost(
                iboost, X, y, sample_weight, sample_quality, random_state, fit_params
            )

            # Early termination
            if sample_weight is None:
                break
            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            # Stop if error is zero
            if estimator_error == 0:
                break

            sample_weight_sum = np.sum(sample_weight)

            if not np.isfinite(sample_weight_sum):
                warnings.warn(
                    "Sample weights have reached infinite values,"
                    f" at iteration {iboost}, causing overflow. "
                    "Iterations stopped. Try lowering the learning rate.",
                    stacklevel=2,
                )
                break

            # Stop if the sum of sample weights has become non-positive
            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
                # Normalize
                sample_weight /= sample_weight_sum

        return self

    def _validate_estimator(self):
        """Check the estimator and set the estimator_ attribute."""
        super(BaseWeightBoosting, self)._validate_estimator(default=LinearSVC())

        if not has_fit_parameter(self.estimator_, "sample_weight"):
            raise ValueError(
                "%s doesn't support sample_weight." % self.estimator_.__class__.__name__
            )

    def _boost(
        self, iboost, X, y, sample_weight, sample_quality, random_state, fit_params
    ):
        """Implement a single boost using the Transfert SAMME discrete algorithm."""

        estimator = self._make_estimator(random_state=random_state)

        estimator.fit(X, y, sample_weight=sample_weight, **fit_params)

        y_predict = estimator.predict(X)

        if iboost == 0:
            self.classes_ = getattr(estimator, "classes_", None)
            self.n_classes_ = len(self.classes_)

        # Instances incorrectly classified
        incorrect = y_predict != y

        # Error fraction on trusted dataset
        estimator_error = np.mean(
            np.average(
                incorrect[sample_quality == 1],
                weights=sample_weight[sample_quality == 1],
                axis=0,
            )
        )

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1.0, 0.0

        # Stop if the error is at least as bad as random guessing
        if estimator_error >= 1 - (1 / self.n_classes_):
            # Worse than random guessing
            self.estimators_.pop(-1)
            if len(self.estimators_) == 0:
                raise ValueError(
                    "BaseClassifier in TrAdaBoostClassifier "
                    "ensemble is worse than random, ensemble "
                    "can not be fit."
                )
            return None, None, None

        beta = (estimator_error / (1 - estimator_error)) * (1 / (self.n_classes_ - 1))

        n_samples_untrusted = np.size(sample_quality) - np.count_nonzero(sample_quality)
        n_samples_untrusted = max(n_samples_untrusted, 1)
        beta_const = 1 / (
            1 + math.sqrt(2 * math.log(n_samples_untrusted) / self.n_estimators)
        )

        # Estimator weight
        estimator_weight = self.learning_rate * math.log(1 / beta)

        # Only boost the weights if will fit again
        if not iboost == self.n_estimators - 1:
            # Boost trusted weights using AdaBoost SAMME alg
            sample_weight *= np.power(
                beta,
                -self.learning_rate
                * incorrect
                * (sample_weight > 0)
                * (sample_quality == 1),
            )
            # Boost untrusted weights using WMA alg
            sample_weight *= np.power(
                beta_const,
                self.learning_rate
                * incorrect
                * (sample_weight > 0)
                * (sample_quality == 0),
            )
            # Multi-class weight drift correction
            sample_weight[sample_quality == 0] *= (
                1 - estimator_error
            ) + estimator_error * math.pow(beta, -self.learning_rate)

        return (
            sample_weight,
            estimator_weight,
            estimator_error,
        )

    def _more_tags(self):
        return {"_xfail_checks": {"check_dict_unchanged": "tofix"}}
