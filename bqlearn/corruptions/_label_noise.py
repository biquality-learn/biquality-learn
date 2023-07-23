import warnings
from functools import partial
from inspect import signature

import numpy as np
from scipy.special import softmax
from scipy.stats import entropy, norm, truncnorm
from sklearn.dummy import check_random_state
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.utils.multiclass import type_of_target, unique_labels
from sklearn.utils.validation import _num_features, _num_samples

from ..utils import categorical
from .noise_matrices import NOISE_MATRIX_FUNCTIONS


def make_label_noise(
    y, noise_matrix="uniform", *, noise_ratio=0.5, random_state=None, labels=None
):
    """Corrupt the labels given a noise transition matrix.

    Parameters
    ----------
    y : array-like of shape (n_samples, )
        The targets.

    noise_matrix : str or array-like of shape (n_classes, n_classes), default="uniform"
        The matrix representing probabilities transition between clean labels
        and noisy labels. If noise_matrix is a string, it must be one of the metrics
        in noise_matrices.NOISE_MATRIX_FUNCTIONS.

    noise_ratio : float, default=0.5
        The ratio of noise. Must be between 0 and 1.
        Not used if noise_matrix is an array-like.

    random_state : int or RandomState, default=None
        Controls the noise matrix construction.

    labels : array-like of shape (n_classes), default=None
        List of labels to index the matrix. This may be used to reorder
        or select a subset of labels.
        If ``None`` is given, those that appear at least once in ``y``
        and are used in sorted order.

    Returns
    -------
    y_corrupt : ndarray of shape (n_samples,)
        The corrupted targets.
    """

    y_type = type_of_target(y)
    if y_type not in ("binary", "multiclass"):
        raise ValueError("%s is not supported" % y_type)

    if labels is None:
        labels = unique_labels(y)

    le = LabelEncoder().fit(labels)
    y = le.transform(y)
    classes = le.classes_
    n_classes = len(classes)

    if not np.all(classes == labels):
        warnings.warn(
            f"Labels passed were {labels}. But this function "
            "assumes labels are ordered lexicographically. "
            "Ensure that labels in y are ordered as "
            f"{classes}.",
            UserWarning,
        )

    if hasattr(noise_matrix, "shape"):
        noise_matrix_ = noise_matrix

    elif noise_matrix in NOISE_MATRIX_FUNCTIONS:
        noise_matrix_func = NOISE_MATRIX_FUNCTIONS[noise_matrix]
        if "random_state" in signature(noise_matrix_func).parameters:
            noise_matrix_func = partial(noise_matrix_func, random_state=random_state)

        noise_matrix_ = noise_matrix_func(n_classes, noise_ratio)

    else:
        raise ValueError("Unknown matrix %r" % noise_matrix)

    probabilities = noise_matrix_[y]

    return le.inverse_transform(categorical(probabilities, random_state=random_state))


def make_instance_dependent_label_noise(
    noise_prob,
    y,
    noise_matrix="uniform",
    *,
    random_state=None,
    labels=None,
):
    """Corrupt the labels given a noise transition matrix
    and a noise probability function.

    Parameters
    ----------
    noise_prob : array-like of shape (n_samples, )
        The noise probabilities.

    y : array-like of shape (n_samples, )
        The targets.

    noise_matrix : str or callable, default="uniform"
        The matrix representing probabilities transition between clean labels
        and noisy labels. If noise_matrix is a string, it must be one of the metrics
        in noise_matrices.NOISE_MATRIX_FUNCTIONS. If noise_matrix is a callable,
        it should take a number of classes, a noise probability, and optionally
        a random state as the input and outputs a numpy array
        of shape (n_classes, n_classes).

    random_state : int or RandomState, default=None
        Controls the noise matrix construction.

    labels : array-like of shape (n_classes), default=None
        List of labels to index the matrix. This may be used to reorder
        or select a subset of labels.
        If ``None`` is given, those that appear at least once in ``y``
        and are used in sorted order.

    Returns
    -------
    y_corrupt : ndarray of shape (n_samples,)
        The corrupted targets.
    """

    y_type = type_of_target(y)
    if y_type not in ("binary", "multiclass"):
        raise ValueError("%s is not supported" % y_type)

    if labels is None:
        labels = unique_labels(y)

    le = LabelEncoder().fit(labels)
    y = le.transform(y)
    classes = le.classes_
    n_classes = len(classes)

    n_samples = _num_samples(y)

    if not np.all(classes == labels):
        warnings.warn(
            f"Labels passed were {labels}. But this function "
            "assumes labels are ordered lexicographically. "
            "Ensure that labels in y are ordered as "
            f"{classes}.",
            UserWarning,
        )

    if callable(noise_matrix):
        noise_matrix_func = noise_matrix

    elif noise_matrix in NOISE_MATRIX_FUNCTIONS:
        noise_matrix_func = NOISE_MATRIX_FUNCTIONS[noise_matrix]
    else:
        raise ValueError("Unknown matrix %r" % noise_matrix)

    if "random_state" in signature(noise_matrix_func).parameters:
        noise_matrix_func = partial(noise_matrix_func, random_state=random_state)

    noise_prob = np.clip(noise_prob, 0, 1)

    # TODO: find a better way
    probabilities = np.empty((n_samples, n_classes))
    for i, p in enumerate(noise_prob):
        probabilities[i, :] = noise_matrix_func(n_classes, p)[y[i]]

    return le.inverse_transform(categorical(probabilities, random_state=random_state))


def uncertainty_noise_probability(
    X, estimator, uncertainty="uncertainty", noise_ratio=0.5, random_state=None
):
    """Get a probability of a sample to be noisy given an uncertainty function
    according to [1]_.

    Valid uncertainty functions are:
        ['uncertainty', 'margin', 'entropy', 'density']

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The samples.

    estimator : object
        The fitted estimator used to compute the noise probability.
        Must have attribute predict_proba or score_samples.

    uncertainty : {'uncertainty', 'margin', 'entropy', 'density'}, default='uncertainty'
        Uncertainty function.
        'density' is only available for estimators with score_samples attribute.

    noise_ratio : float, default=0.5
        The ratio of noise. Must be between 0 and 1.

    random_state : int or RandomState, default=None
        Controls the noisy samples selection.

    Returns
    -------
    noise_probabilities : array-like of shape (n_samples, )
        The noise probabilities.

    References
    ----------
    .. [1] P. Nodet, V. Lemaire, A. Bondu, A. Cornuéjols,\
        "Importance Reweighting for Biquality Learning", IJCNN, 2021.
    """

    if uncertainty in ("uncertainty", "margin", "entropy"):
        if not hasattr(estimator, "predict_proba"):
            raise ValueError(
                "estimator must have attribute predict_proba with uncertainty %s"
                % uncertainty
            )

        probabilities = estimator.predict_proba(X)

        if uncertainty == "uncertainty":
            noise_probabilities = 1 - probabilities.max(axis=1)
        elif uncertainty == "margin":
            part = np.partition(-probabilities, 1, axis=1)
            margin = -part[:, 0] + part[:, 1]
            noise_probabilities = 1 - margin
        elif uncertainty == "entropy":
            noise_probabilities = entropy(probabilities.T)

    elif uncertainty == "density":
        if not hasattr(estimator, "score_samples"):
            raise ValueError(
                "estimator must have attribute score_samples with uncertainty %s"
                % uncertainty
            )

        noise_probabilities = np.exp(-estimator.score_samples(X))

    else:
        raise ValueError("%s uncertainty function is not supported" % uncertainty)

    n_samples = _num_samples(X)

    rng = check_random_state(random_state)
    idx_noisy = rng.choice(
        np.arange(n_samples),
        int(n_samples * noise_ratio),
        p=noise_probabilities / noise_probabilities.sum(),
        replace=False,
    )

    is_noisy = np.zeros(n_samples)
    is_noisy[idx_noisy] = 1.0

    return is_noisy


def noisy_leaves_probability(
    X,
    y,
    *,
    noise_ratio=0.5,
    purity="random",
    min_samples_leaf=1,
    random_state=None,
):
    """Noisify some leaves of a decision tree learn on the input dataset.
    These leaves can be chosen completly at random or prioritizing them by their purity.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The samples.

    y : array-like of shape (n_samples, )
        The targets.

    noise_ratio : float, default=0.5
        The ratio of noise. Must be between 0 and 1.

    purity: {'random', 'ascending', 'descending'}, default='random'
        Choose leaves completly at random or prioritize pure/impure leaves.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
        `ceil(min_samples_leaf * n_samples)` are the minimum
        number of samples for each node.

    random_state : int or RandomState, default=None
        Controls the training of the DecisionTreeClassifier and
        the noisy leaves selection.

    Returns
    -------
    noise_probabilities : array-like of shape (n_samples, )
        The noise probabilities.
    """

    tree = DecisionTreeClassifier(
        min_samples_leaf=min_samples_leaf, random_state=random_state
    )
    tree.fit(X, y)

    X_leaves = tree.apply(X)
    leaves = np.unique(X_leaves)

    leaves = shuffle(leaves, random_state=random_state)

    if purity == "random":
        sorted_leaves = leaves
    elif purity in ("ascending", "descending"):
        impurity = tree.tree_.impurity[leaves]
        if purity == "descending":
            impurity = -impurity
        sorted_leaves = leaves[np.argsort(impurity)]
    else:
        raise ValueError("%s purity is not supported" % purity)

    n_samples = _num_samples(X)
    n_selected_leaves = (
        np.cumsum(tree.tree_.n_node_samples[sorted_leaves]) >= noise_ratio * n_samples
    ).argmax()
    selected_leaves = sorted_leaves[:n_selected_leaves]

    return np.isin(X_leaves, selected_leaves).astype(float)


def make_feature_dependent_label_noise(
    X, y, *, noise_ratio=0.5, random_state=None, labels=None
):
    """Corrupt the labels using a noise distribution model by
    a random linear projection from the features to the labels [1]_.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The samples.

    y : array-like of shape (n_samples, )
        The targets.

    noise_ratio : float, default=0.5
        The ratio of noise. Must be between 0 and 1.

    random_state : int or RandomState, default=None
        Controls the noise matrix construction.

    labels : array-like of shape (n_classes), default=None
        List of labels to index the matrix. This may be used to reorder
        or select a subset of labels.
        If ``None`` is given, those that appear at least once in ``y``
        and are used in sorted order.

    Returns
    -------
    y_corrupt : ndarray of shape (n_samples,)
        The corrupted targets.

    References
    ----------
    .. [1]  Xia, X., Liu, T., Han, B., Wang, N., Gong, M., Liu, H., Sugiyama, M.,\
        "Part-dependent label noise: Towards instance-dependent label noise",\
        NeurIPS 2020.
    """
    y_type = type_of_target(y)
    if y_type not in ("binary", "multiclass"):
        raise ValueError("%s is not supported" % y_type)

    if labels is None:
        labels = unique_labels(y)

    le = LabelEncoder().fit(labels)
    y = le.transform(y)

    classes = le.classes_
    n_classes = len(classes)

    n_samples = _num_samples(X)
    n_features = _num_features(X)

    noise_ratios = truncnorm(0, 1, loc=noise_ratio, scale=0.1).rvs(
        (n_samples, 1), random_state=random_state
    )
    noise_distribution = norm(loc=0, scale=1).rvs(
        (n_features, n_classes), random_state=random_state
    )

    noise_probabilities = X @ noise_distribution
    np.put_along_axis(noise_probabilities, y.reshape(-1, 1), -np.inf, axis=1)
    noise_probabilities = noise_ratios * softmax(noise_probabilities, axis=1)
    np.put_along_axis(noise_probabilities, y.reshape(-1, 1), 1 - noise_ratios, axis=1)

    return le.inverse_transform(
        categorical(noise_probabilities, random_state=random_state)
    )
