"""Metrics for Biquality Data and Biquality Learning"""

import warnings

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_array, check_consistent_length
from sklearn.utils.multiclass import type_of_target, unique_labels
from sklearn.utils.validation import check_scalar

__all__ = [
    "gold_transition_matrix",
    "anchor_transition_matrix",
    "iterative_anchor_transition_matrix",
]


# TODO: replace input validation
# when https://github.com/scikit-learn/scikit-learn/pull/22046 is merged
def gold_transition_matrix(y_true, y_prob, labels=None):
    """Compute the gold transition matrix [1]_.

    It computes the average predictions of a model learned on untrusted data
    on the trusted dataset per class:

    .. math::

        \hat{T}_{(i,*)} = \\frac{1}{|D^i_T|}\sum_{x_i \in D^i_T}f_U(x_i)

    where:

    .. math::

        \\forall k \in [\![1,K]\!], D^k_T = \{\\forall (x,y) \in D_T  \mid  y=k\}

    and $K$ is the number of class.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_prob : array-like of shape (n_samples, n_classes)
        Predicted probabilities, as returned by a classifier's
        predict_proba method.

    labels : array-like of shape (n_classes), default=None
        List of labels to index the matrix. This may be used to reorder
        or select a subset of labels.
        If ``None`` is given, those that appear at least once
        in ``y_true`` or ``y_prob`` are used in sorted order.

    Returns
    -------
    C : ndarray of shape (n_classes, n_classes)
        Gold transition matrix whose i-th row and j-th
        column entry indicates the probability of
        samples with true label being i-th class
        to be corrupted to a label being the j-th class.

    References
    ----------
    .. [1] D. Hendrycks, "Using Trusted Data to Train Deep Networks on\
        Labels Corrupted by Severe Noise", 2019
    """
    y_prob = check_array(y_prob)
    check_consistent_length(y_true, y_prob)

    y_type = type_of_target(y_true)
    if y_type not in ("binary", "multiclass"):
        raise ValueError("%s is not supported" % y_type)

    if y_prob.max() > 1:
        raise ValueError("y_prob contains values greater than 1.")
    if y_prob.min() < 0:
        raise ValueError("y_prob contains values less than 0.")
    if not np.allclose(y_prob.sum(axis=1), 1):
        raise ValueError("y_prob needs to sum up to 1.0 over classes.")

    if labels is None:
        labels = unique_labels(y_true)

    le = LabelEncoder().fit(labels)
    y_true = le.transform(y_true)
    n_classes = len(le.classes_)

    if not np.all(le.classes_ == labels):
        warnings.warn(
            f"Labels passed were {labels}. But this function "
            "assumes labels are ordered lexicographically. "
            "Ensure that labels in y_prob are ordered as "
            f"{le.classes_}.",
            UserWarning,
        )

    if y_prob.shape[1] != n_classes:
        raise ValueError(
            f"y_prob has {y_prob.shape[1]} classes but should have {n_classes}"
        )

    _dtype = np.float64

    tm = np.zeros((n_classes, n_classes), dtype=_dtype)
    np.add.at(tm, y_true, y_prob)
    norm = np.bincount(y_true, minlength=n_classes)

    mask = ~tm.any(axis=1)
    tm[mask] = np.ones(n_classes, dtype=_dtype)
    norm[mask] = n_classes

    tm /= norm[:, np.newaxis]

    return tm


def _anchor_points(y_prob, quantile):
    """compute anchor points of a dataset, modifies the input array inplace"""

    outliers = np.quantile(y_prob, q=quantile, axis=0)
    y_prob[y_prob > outliers] = 0
    return np.argmax(y_prob, axis=0)


def anchor_transition_matrix(y_prob, quantile=0.97, anchor_idx=None):
    """Compute the anchor transition matrix [1]_.

    It uses anchor points :math:`A` as trustful points in a unlabelled dataset.

    .. math::

        \\forall i \in [\![1,K]\!], A_i =
        \operatorname*{argmax}_{x \in D} \mathbb{P}(Y=i|X=x)

    Then it uses predictions of a model learned on untrusted data to estimate
    the transition matrix.

    .. math::

        \\forall (i,j) \in [\![1,K]\!]^2, \hat{T}_{(i,j)}
        = \mathbb{P}(\\tilde{Y}=j|X=A_i)

    Parameters
    ----------
    y_prob : array-like of shape (n_samples, n_classes)
        Predicted probabilities, as returned by a classifier's
        predict_proba method.

    quantile : float, default=0.97
        Quantile used to select the anchor points.
        It filters out outlier points with high predicted probabilities.

    anchor_idx : array-like of shape (n_classes), default=None
        Anchor points indices.
        If not None, use provided anchor points instead of computing them.

    Returns
    -------
    C : ndarray of shape (n_classes, n_classes)
        Anchor transition matrix whose i-th row and j-th
        column entry indicates the probability of
        samples with true label being i-th class
        to be corrupted to a label being the j-th class.

    References
    ----------
    .. [1] G. Patrini, "Making Deep Neural Networks Robust to Label Noise:
           a Loss Correction Approach", 2017
    """
    y_prob = check_array(y_prob)

    if y_prob.max() > 1:
        raise ValueError("y_prob contains values greater than 1.")
    if y_prob.min() < 0:
        raise ValueError("y_prob contains values less than 0.")
    if not np.allclose(y_prob.sum(axis=1), 1):
        raise ValueError("y_prob needs to sum up to 1.0 over classes.")

    check_scalar(quantile, "quantile", float, min_val=0, max_val=1)

    n_classes = y_prob.shape[1]

    if anchor_idx is None:
        anchor_idx = _anchor_points(np.copy(y_prob), quantile)
    else:
        anchor_idx = check_array(anchor_idx, ensure_2d=False)

    ii, jj = np.meshgrid(anchor_idx, np.arange(0, n_classes), indexing="ij")

    return y_prob[ii, jj]


def iterative_anchor_transition_matrix(y_prob, quantile=0.97, n_iter=100):
    """Compute a transition matrix based on an iterative algorithm
    using anchor points [1]_.

    Parameters
    ----------
    y_prob : array-like of shape (n_samples, n_classes)
        Predicted probabilities, as returned by a classifier's
        predict_proba method.

    quantile : float, default=0.97
        Quantile used to select the anchor points.
        It filters out outlier points with high predicted probabilities.

    n_iter : int, default=100
        Number of time an enhanced anchor transition matrix is computed.

    Returns
    -------
    C : ndarray of shape (n_classes, n_classes)
        Anchor transition matrix whose i-th row and j-th
        column entry indicates the probability of
        samples with true label being i-th class
        to be corrupted to a label being the j-th class.

    References
    ----------
    .. [1] M. Zhang, J. Lee, and S. Agarwal.
        "Learning from noisy labels with no change to the training process.",
        ICML, 2021.
    """

    y_prob = check_array(y_prob)

    if y_prob.max() > 1:
        raise ValueError("y_prob contains values greater than 1.")
    if y_prob.min() < 0:
        raise ValueError("y_prob contains values less than 0.")
    if not np.allclose(y_prob.sum(axis=1), 1):
        raise ValueError("y_prob needs to sum up to 1.0 over classes.")

    check_scalar(quantile, "quantile", (float, int), min_val=0, max_val=1)
    check_scalar(n_iter, "n_iter", int, min_val=1)

    n_classes = y_prob.shape[1]

    transition_matrices = []
    transition_matrix = np.identity(n_classes)
    transition_matrices.append(transition_matrix)

    diffs = []

    for _ in range(n_iter):
        # Non invertible transition matrix
        if np.linalg.matrix_rank(transition_matrices[-1]) != n_classes:
            break

        y_score = y_prob @ np.linalg.inv(transition_matrices[-1])
        anchor_idx = _anchor_points(y_score, quantile)

        transition_matrix = anchor_transition_matrix(y_prob, anchor_idx=anchor_idx)
        diff = np.linalg.norm(transition_matrix - transition_matrices[-1], ord="fro")

        transition_matrices.append(transition_matrix)
        diffs.append(diff)

    return transition_matrices[np.argmin(diffs) + 1]
