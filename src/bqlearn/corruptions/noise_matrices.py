"""Noise matrices"""

from functools import partial

import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_scalar

__all__ = ["uniform_noise_matrix", "flip_noise_matrix", "background_noise_matrix"]


def uniform_noise_matrix(n_classes, noise_ratio, dtype=float):
    """Uniform noise matrix

    Parameters
    ----------
    n_classes : int
        The number of classes.

    noise_ratio : float
        The ratio of noise. Must be between 0 and 1.

    dtype : data-type, default=None
        Overrides the data type of the result.

    Returns
    -------
    matrix : ndarray of shape (n_classes, n_classes)
        Returns the noise matrix.
    """

    check_scalar(n_classes, "n_classes", int, min_val=1)
    check_scalar(noise_ratio, "noise_ratio", (float, int), min_val=0, max_val=1)

    return noise_ratio * np.full((n_classes, n_classes), 1 / n_classes, dtype=dtype) + (
        1 - noise_ratio
    ) * np.identity(n_classes, dtype=dtype)


def flip_noise_matrix(
    n_classes, noise_ratio, permutation=False, dtype=float, random_state=None
):
    """Flip noise matrix.

    Parameters
    ----------
    n_classes : int
        The number of classes.

    noise_ratio : float
        The ratio of noise. Must be between 0 and 1.

    permutation : boolean, default=False
        If True, construct a permutation matrix, such that two different classes
        can't flip to the same noisy class.

    dtype : data-type, default=None
        Overrides the data type of the result.

    random_state : int or RandomState, default=None
        Controls the flip direction.

    Returns
    -------
    matrix : ndarray of shape (n_classes, n_classes)
        Returns the noise matrix.
    """

    check_scalar(n_classes, "n_classes", int, min_val=1)
    check_scalar(noise_ratio, "noise_ratio", (float, int), min_val=0, max_val=1)

    rng = check_random_state(random_state)

    corruption_matrix = np.zeros((n_classes, n_classes), dtype=dtype)

    classes = np.arange(n_classes)
    p = np.ones(n_classes)

    for i in range(n_classes):
        p_i = np.copy(p)

        if p_i.sum() > 1:
            p_i[i] = 0

        flipto_i = rng.choice(classes, p=p_i / p_i.sum())
        corruption_matrix[i][flipto_i] = 1

        if permutation:
            p[flipto_i] = 0

    # permutation algorithm could lead to a case where the last class
    # is associated with itself, we fix that by swapping the two last rows
    # of the noise matrix
    if permutation and corruption_matrix[-1, -1] == 1:
        corruption_matrix[[-2, -1], :] = corruption_matrix[[-1, -2], :]

    return noise_ratio * corruption_matrix + (1 - noise_ratio) * np.identity(
        n_classes, dtype=dtype
    )


def background_noise_matrix(n_classes, noise_ratio, dtype=float, random_state=None):
    """Background noise matrix.

    Parameters
    ----------
    n_classes : int
        The number of classes.

    noise_ratio : float
        The ratio of noise. Must be between 0 and 1.

    dtype : data-type, default=None
        Overrides the data type of the result.

    random_state : int or RandomState, default=None
        Controls the background class.

    Returns
    -------
    matrix : ndarray of shape (n_classes, n_classes)
        Returns the noise matrix.
    """

    check_scalar(n_classes, "n_classes", int, min_val=1)
    check_scalar(noise_ratio, "noise_ratio", (float, int), min_val=0, max_val=1)

    random_state = check_random_state(random_state)

    background_class = random_state.randint(n_classes, size=1)[0]

    corruption_matrix = np.zeros((n_classes, n_classes), dtype=dtype)
    corruption_matrix[:, background_class] = 1.0

    return noise_ratio * corruption_matrix + (1 - noise_ratio) * np.identity(
        n_classes, dtype=dtype
    )


NOISE_MATRIX_FUNCTIONS = {
    "uniform": uniform_noise_matrix,
    "flip": flip_noise_matrix,
    "permutation": partial(flip_noise_matrix, permutation=True),
    "background": background_noise_matrix,
}
