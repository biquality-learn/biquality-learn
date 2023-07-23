import warnings

import numpy as np
from joblib import delayed, Parallel
from sklearn import clone
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import _safe_indexing, check_consistent_length, check_random_state
from sklearn.utils.multiclass import type_of_target, unique_labels
from sklearn.utils.validation import _num_samples, check_scalar, indexable


def make_imbalance(
    y,
    *arrays,
    majority_ratio=1.0,
    imbalance_distribution="step",
    minority_class_fraction=0.5,
    random_state=None,
    labels=None,
):
    """
    Create class imbalance in a multi class scenario according to [1]_.

    It selects a fraction of all class to be considered as the minority group
    according to `minority_class_fraction` and going to subsample it given
    `majority_ratio` when `imbalance_distribution='step'`.

    If `imbalance_distribution='linear'`, it creates imbalance between all classes by
    decreasing linearly the ratio of subsampling when iterating through classes
    according to `majority_ratio`.

    Parameters
    ----------
    y : array-like of shape (n_samples, )
        The targets.

    *arrays: sequence of indexables with length / shape[0] equals to n_samples
        Allowed inputs are lists, numpy arrays,
        scipy-sparse matrices or pandas dataframes.

    majority_ratio : float, default = 1.0
        Ratio between number of samples in majority classes and
        number of samples in minority classes.

    imbalance_distribution : {'step', 'linear'}, default='step'
        Imbalance distribution.

    minority_class_fraction : float, default = 0.5
        Fraction of classes considered as minority classes. Only used
        when `imbalance_distribution='step'`.

    random_state : int or RandomState, default=None
        Controls the randomness of the subsampling procedure.

    labels : array-like of shape (n_classes), default=None
        List of labels to index the matrix. This may be used to reorder
        or select a subset of labels.
        If ``None`` is given, those that appear at least once in ``y``
        and are used in sorted order.

    Returns
    -------
    y_imbalanced : array-like of shape (n_samples_new)
        The array containing the imbalanced data.

    *arrays_imbalanced : list, length=len(arrays)
        The corresponding imbalanced arrays.

    References
    ----------
    .. [1] Mateusz Buda, et al. "A systematic study of the class imbalance problem\
        in convolutional neural networks." Neural Networks, 106:249-259, 2018.
    """

    rng = check_random_state(random_state)

    arrays = indexable(*arrays)

    check_consistent_length(y, *arrays)

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

    check_scalar(
        majority_ratio,
        "majority_ratio",
        (float, int),
        min_val=1,
        include_boundaries="left",
    )
    check_scalar(
        minority_class_fraction,
        "minority_class_fraction",
        (float, int),
        min_val=0,
        max_val=1,
        include_boundaries="neither",
    )

    n_samples_per_class = np.bincount(y)
    argsort_classes = np.argsort(n_samples_per_class)

    if imbalance_distribution == "linear":
        sampling_probability = np.linspace(1 / majority_ratio, 1.0, n_classes)
        sampling_probability = sampling_probability[argsort_classes]

    elif imbalance_distribution == "step":
        sampling_probability = np.ones(n_classes)
        n_minority_classes = round(n_classes * minority_class_fraction)
        sampling_probability[argsort_classes[:n_minority_classes]] /= majority_ratio

    else:
        raise ValueError(
            f"Unsupported imbalance distribution : {imbalance_distribution}"
        )

    acc = np.empty((0,), dtype=int)

    for i in range(n_classes):
        idx = rng.choice(
            range(np.count_nonzero(y == i)),
            size=int(n_samples_per_class[i] * sampling_probability[i]),
            replace=False,
        )

        acc = np.concatenate(
            (
                acc,
                np.flatnonzero(y == i)[idx],
            ),
            axis=0,
        )

    y = le.inverse_transform(y)

    if len(arrays) == 0:
        return _safe_indexing(y, acc)
    else:
        return (
            _safe_indexing(y, acc),
            *[_safe_indexing(array, acc) for array in arrays],
        )


def make_cluster_imbalance(
    X,
    y,
    *arrays,
    per_class_n_clusters=3,
    majority_ratio=1.0,
    imbalance_distribution="step",
    minority_class_fraction=0.5,
    random_state=None,
    n_jobs=None,
):
    """
    Create per-class cluster imbalance in a multi class scenario according to [1]_.

    Learns a :mod:`sklearn.cluster.KMeans` clustering once per class and creates
    class imbalance based on the cluster labels.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The samples.

    y : array-like of shape (n_samples, )
        The targets.

    *arrays: sequence of indexables with length / shape[0] equals to n_samples
        Allowed inputs are lists, numpy arrays,
        scipy-sparse matrices or pandas dataframes.

    per_class_n_clusters : dict or int, default = 3
        The number of clusters are associated with classes in the form
        ``{class_label: n_cluster}`` for the KMeans algorithm.

        If an ``int``, then the same number of clusters is used for all classes.

    majority_ratio : float, default = 1.0
        Ratio between number of samples in majority classes and
        number of samples in minority classes.

    imbalance_distribution : {'step', 'linear'}, default='step'
        Imbalance distribution.

    minority_class_fraction : float, default = 0.5
        Fraction of classes considered as minority classes. Only used
        when `imbalance_distribution='step'`.

    random_state : int or RandomState, default=None
        Controls the randomness of the KMeans clustering.

    n_jobs : int, default=None
        The number of jobs to use for the computation. This parallelize the
        training of :mod:`sklearn.cluster.KMeans` per class.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Returns
    -------
    X_imbalanced : array-like of shape (n_samples_new, n_features)
        The array containing the imbalanced data.

    y_imbalanced : array-like of shape (n_samples_new)
        The corresponding label of X_imbalanced.

    *arrays_imbalanced : list, length=len(arrays)
        The corresponding imbalanced arrays.

    References
    ----------
    .. [1] P. Nodet, V. Lemaire, A. Bondu, A. Cornuéjols,\
        "Design of Algorithms Dealing with Closed-Set Distribution Shifts", 2023.
    """

    random_state = check_random_state(random_state)

    check_consistent_length(X, y)

    y_type = type_of_target(y)
    if y_type not in ("binary", "multiclass"):
        raise ValueError("%s is not supported" % y_type)

    classes = unique_labels(y)
    n_classes = len(classes)

    n_samples = _num_samples(X)

    if isinstance(per_class_n_clusters, dict):
        for value in per_class_n_clusters.values():
            if not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"""per_class_n_clusters sould only contain strictly positive
                    integers, was : {per_class_n_clusters}"""
                )
        if (list(per_class_n_clusters.keys()) != classes).any():
            raise ValueError(
                f"""per_class_n_clusters keys are different of class labels,
                    was : {per_class_n_clusters.keys()} instead of : {classes}"""
            )
    elif isinstance(per_class_n_clusters, int):
        if per_class_n_clusters > 0:
            per_class_n_clusters = dict(
                zip(classes, [per_class_n_clusters] * n_classes)
            )
        else:
            raise ValueError(
                f"per_class_n_clusters={per_class_n_clusters} should be > 0"
            )
    else:
        raise ValueError(
            f"per_class_n_clusters is not supported, was : {per_class_n_clusters}"
        )

    Xs = [_safe_indexing(X, y == c) for c in classes]

    index = np.arange(n_samples)
    indexes = [_safe_indexing(index, y == c) for c in classes]

    k_means = [
        clone(KMeans(n_clusters=per_class_n_clusters[c], random_state=random_state))
        for c in classes
    ]

    def fit_predict(k_means, X):
        return k_means.fit_predict(X)

    class_cluster_labels = Parallel(n_jobs=n_jobs)(
        delayed(fit_predict)(k_means_i, X_i) for (k_means_i, X_i) in zip(k_means, Xs)
    )

    indexes_imbalanced = []

    for index_i, cluster_labels in zip(indexes, class_cluster_labels):
        _, index_imbalanced = make_imbalance(
            cluster_labels,
            index_i,
            majority_ratio=majority_ratio,
            imbalance_distribution=imbalance_distribution,
            minority_class_fraction=minority_class_fraction,
            random_state=random_state,
        )
        indexes_imbalanced.append(index_imbalanced)

    indexes_imbalanced = np.hstack(indexes_imbalanced)

    X_imbalanced = _safe_indexing(X, indexes_imbalanced)
    y_imbalanced = _safe_indexing(y, indexes_imbalanced)
    arrays_imbalanced = [_safe_indexing(array, indexes_imbalanced) for array in arrays]

    return X_imbalanced, y_imbalanced, *arrays_imbalanced
