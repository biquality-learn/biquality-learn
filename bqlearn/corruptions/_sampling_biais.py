import math

import numpy as np
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.utils import (
    _safe_indexing,
    check_consistent_length,
    check_random_state,
    check_scalar,
    indexable,
)
from sklearn.utils.validation import _num_samples


def make_sampling_biais(
    X,
    *arrays,
    a=3,
    b=8,
    random_state=None,
):
    """
    Synthetic covariate shift by creating a sampling biais using
    the first axis of a PCA learned of the input features [1]_.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The samples.

    *arrays: sequence of indexables with length / shape[0] equals to n_samples
        Allowed inputs are lists, numpy arrays,
        scipy-sparse matrices or pandas dataframes.

    a : float, default = 3.0

    b : float, default = 8.0

    random_state : int or RandomState, default=None
        Controls the randomness of the PCA.

    Returns
    -------
    X_corrupted : ndarray of shape (n_samples, n_features)
        The corrupted samples.

    *arrays_imbalanced : list, length=len(arrays)
        The corresponding imbalanced arrays.

    References
    ----------
    .. [1] Gretton, Arthur, et al. "Covariate shift by kernel mean matching."
        Dataset shift in machine learning 3.4 (2009): 5.
    """

    check_consistent_length(X, *arrays)

    n_samples = _num_samples(X)

    pca = PCA(n_components=1, random_state=random_state).fit(X)

    X_first_axis = pca.transform(X).ravel()

    check_scalar(a, "a", (float, int), min_val=0, include_boundaries="neither")
    check_scalar(b, "b", (float, int), min_val=0, include_boundaries="neither")

    mean = np.mean(X_first_axis)
    min = np.min(X_first_axis)

    loc = min + (mean - min) / a
    eps = np.finfo(float).eps
    scale = math.sqrt((mean - min) / b) + eps

    sampling_prob = norm.pdf(X_first_axis, loc=loc, scale=scale)

    rng = check_random_state(random_state)
    idx = rng.choice(
        np.arange(n_samples), n_samples, p=sampling_prob / sampling_prob.sum()
    )

    arrays = indexable(*arrays)

    return [_safe_indexing(X, idx)] + [_safe_indexing(a, idx) for a in arrays]
