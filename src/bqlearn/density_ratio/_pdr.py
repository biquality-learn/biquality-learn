"""Probability Density Ratio Estimation for Biquality Learning."""

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import check_pairwise_arrays
from sklearn.utils.validation import _num_samples

from ..utils import safe_sparse_vstack
from ._idr import IDR
from ._kdr import KDR


def pdr(X, Y, estimator, method="probabilities"):
    """Probabilistic Density Ratio [1]_.

    Parameters
    ----------
    X : array-like of shape (n_samples_X, n_features)
        A feature array.

    Y : array-like of shape (n_samples_Y, n_features)
        A second feature array.

    estimator : object
        Probabilistic binary classifier.

    method : {'probabilities', 'odds'}, default='probabilities'
        Method to compute the ratio of conditional probabilities.

    Returns
    ----------
    sample_weights : ndarray, shape (n_samples_X,)
        The weights of the X samples.

    References
    ----------
    .. [1] S. Bickel, M. Bruckner, T. Scheffer,\
        "Discriminative Learning for Differing Training and Test Distributions", 2007
    """

    X, Y = check_pairwise_arrays(X, Y, accept_sparse=True, force_all_finite=False)

    n_samples_X = _num_samples(X)
    n_samples_Y = _num_samples(Y)

    if method not in ("probabilities", "odds"):
        raise ValueError("""Unknown method %s.""" % method)

    if method == "probabilities" and not hasattr(estimator.__class__, "predict_proba"):
        raise ValueError(
            """%s doesn't support predict_proba.""" % estimator.__class__.__name__
        )

    estimator_ = clone(estimator)

    estimator_.fit(
        safe_sparse_vstack((X, Y)),
        np.hstack([np.zeros(n_samples_X), np.ones(n_samples_Y)]),
    )

    prior = n_samples_X / n_samples_Y

    if method == "odds":
        beta_i = np.exp(estimator_.decision_function(X))
    else:
        proba = estimator_.predict_proba(X)
        with np.errstate(divide="ignore", invalid="ignore"):
            beta_i = 1 / proba[:, 0] - 1
            beta_i[proba[:, 0] == 0] = 0

    return prior * beta_i


class KPDR(KDR):
    """A K-Probabilistic Density Ratio Biquality Classifier.

    A KDR using a Probabilistic Classifier [1]_ to reweigth untrusted examples [2]_.

    Parameters
    ----------
    estimator : object
        The final estimator from which the KDR classifier is built.
        Support for sample weighting is required.

    pdr_estimator : object, default=None
        The base estimator from which the weights are estimated thanks to `pdr`.
        If ``None``, then
        the base estimator is :class:`~sklearn.linear_model.LogisticRegression`.

    method: {'odds', 'probabilites'}, default='probabilites'
        Use the odd ratios simplification to avoid the division when computing
        the ratio of conditional probabilities. This method is not adequate for
        estimators using a different link function than the logit.

    n_jobs : int, default=None
        The number of jobs to use for the computation. This parallelize the
        density ratio estimation procedures on all classes.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    estimator_ : classifier
        The final fitted estimator.

    sample_weight_ : ndarray, shape (n_samples,)
        The weights of the examples computed during :meth:`fit`.

    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    n_classes_ : int
        The number of classes.

    References
    ----------
    .. [1] S. Bickel, M. Bruckner, T. Scheffer,\
        "Discriminative Learning for Differing Training and Test Distributions", 2007

    .. [2] P. Nodet, V. Lemaire, A. Bondu, A. Cornuéjols,\
        "Biquality Learning: a Framework to Design Algorithms Dealing\
        with Closed-Set Distribution Shifts.", ECML-PKDD, Machine Learning, 2023.
    """

    def __init__(
        self,
        estimator,
        *,
        pdr_estimator=None,
        method="probabilities",
        n_jobs=None,
    ):
        super().__init__(estimator=estimator, n_jobs=n_jobs)
        self.pdr_estimator = pdr_estimator
        self.method = method

    def _density_ratio(self, X_untrusted_i, X_trusted_i):
        if self.pdr_estimator is None:
            pdr_estimator = LogisticRegression()
        else:
            pdr_estimator = clone(self.pdr_estimator)
        return pdr(X_untrusted_i, X_trusted_i, pdr_estimator, self.method)


class IPDR(IDR):
    """An Iterative Probabilistic Density Ratio Biquality Classifier.

    An IDR using a Probabilistic Classifier [1]_ to reweigth untrusted examples [2]_.

    Parameters
    ----------
    estimator : object
        The estimator from which the IDR classifier is built.
        Support for sample weighting and probability prediction is required.

    n_estimators : int, default=10
        Maximum number of trained estimators on reweighted samples.

    exploit_iterative_learning: boolean, default=False
        If the `estimator` supports iterative learning with `warm_start`,
        exploit it by computing new weights for every epoch when fitting
        `estimator`.

    window: int, default=1
        Number of previous losses used to compute sample weights.

    pdr_estimator : object, default=None
        The base estimator from which the weights are estimated thanks to `pdr`.
        If ``None``, then
        the base estimator is :class:`~sklearn.linear_model.LogisticRegression`.

    method: {'odds', 'probabilites'}, default='probabilites'
        Use the odd ratios simplification to avoid the division when computing
        the ratio of conditional probabilities. This method is not adequate for
        estimators using a different link function than the logit.

    Attributes
    ----------
    estimator_ : classifier
        The final fitted estimator.

    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    n_classes_ : int
        The number of classes.

    References
    ----------
    .. [1] S. Bickel, M. Bruckner, T. Scheffer,\
        "Discriminative Learning for Differing Training and Test Distributions", 2007

    .. [2] Jiang, Lu, et al. "Mentornet: Learning data-driven curriculum for very\
        deep neural networks on corrupted labels."\
        International conference on machine learning. PMLR, 2018.
    """

    def __init__(
        self,
        estimator,
        *,
        n_estimators=10,
        exploit_iterative_learning=False,
        window=1,
        pdr_estimator=None,
        method="probabilities",
    ):
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            exploit_iterative_learning=exploit_iterative_learning,
            window=window,
        )
        self.pdr_estimator = pdr_estimator
        self.method = method

    def _density_ratio(self, loss_untrusted, loss_trusted):
        if self.pdr_estimator is None:
            pdr_estimator = LogisticRegression()
        else:
            pdr_estimator = clone(self.pdr_estimator)
        return pdr(loss_untrusted, loss_trusted, pdr_estimator, self.method)
