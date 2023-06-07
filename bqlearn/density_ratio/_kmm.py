"""A Kernel Mean Matching Estimator for Biquality Learning"""

import math
import warnings

import numpy as np
import scipy.sparse as sp
from joblib import delayed, Parallel
from scs import solve
from sklearn.metrics import pairwise
from sklearn.utils import check_symmetric, gen_batches
from sklearn.utils.validation import _num_samples

from ._idr import IDR
from ._kdr import KDR


def kmm(
    X,
    Y,
    *,
    kernel="rbf",
    kernel_params={},
    B=1000,
    epsilon=None,
    max_iter=1000,
    tol=1e-6,
    n_jobs=None,
):
    """Kernel Mean Matching [1]_.

    Parameters
    ----------
    X : array-like of shape (n_samples_X, n_features)
        A feature array.

    Y : array-like of shape (n_samples_Y, n_features)
        A second feature array.

    kernel : str or callable, default="rbf"
        Kernel mapping used internally. This parameter is directly passed to
        :class:`~sklearn.metrics.pairwise.pairwise_kernel`.
        If `kernel` is a string, it must be one of the metrics
        in `pairwise.PAIRWISE_KERNEL_FUNCTIONS`.
        Alternatively, if `kernel` is a callable function, it is called on
        each pair of instances (rows) and the resulting value recorded. The
        callable should take two rows from X as input and return the
        corresponding kernel value as a single number. This means that
        callables from :mod:`sklearn.metrics.pairwise` are not allowed, as
        they operate on matrices, not single samples. Use the string
        identifying the kernel instead.

    kernel_params : dict, optional (default={})
        Kernel additional parameters

    B: float, optional (default=1000)
        Bounding weights parameter.

    epsilon: float, optional (default=None)
        Constraint parameter.
        If ``None`` epsilon is set to
        ``(np.sqrt(n_samples_X - 1)/np.sqrt(n_samples_X)``.

    max_iter: int, default=100
        Maximum number of iterations. The solver iterates until convergence
        (determined by 'tol') or this number of iterations.

    tol: float, default=1e-4
        Termination criteria dictating the absolute and relative error
        on the primal residual, dual residual and duality gap.

    Returns
    ----------
    sample_weights : ndarray, shape (n_samples_X,)
        The weights of the X samples.

    References
    ----------
    .. [1] Huang, J. and Smola, A. and Gretton, A. and Borgwardt, KM.\
        and Schölkopf, B., "Correcting Sample Selection Bias by Unlabeled Data", 2006
    """

    n_samples_X = _num_samples(X)
    n_samples_Y = _num_samples(Y)

    # Compute Kernel Matrix
    K = pairwise.pairwise_kernels(X, metric=kernel, n_jobs=n_jobs, **kernel_params)
    K = check_symmetric(K, raise_warning=False)
    K = sp.csc_matrix(K)

    # Compute q
    kappa = pairwise.pairwise_kernels(
        X, Y, metric=kernel, n_jobs=n_jobs, **kernel_params
    )
    kappa = n_samples_X / n_samples_Y * np.sum(kappa, axis=1)

    # Get epsilon
    if epsilon is None:
        epsilon = (math.sqrt(n_samples_X) - 1) / math.sqrt(n_samples_X)

    A = sp.vstack(
        [
            np.ones((1, n_samples_X)),
            -np.ones((1, n_samples_X)),
            sp.eye(n_samples_X),
            -sp.eye(n_samples_X),
        ],
        format="csc",
    )
    b = np.hstack(
        [
            n_samples_X * (1 + epsilon),
            n_samples_X * (epsilon - 1),
            B * np.ones(n_samples_X),
            np.zeros(n_samples_X),
        ]
    )

    data = dict(P=K, A=A, b=b, c=-kappa)
    cone = dict(z=0, l=2 * n_samples_X + 2)

    sample_weights_X = solve(
        data,
        cone,
        max_iters=max_iter,
        eps_abs=tol,
        eps_rel=tol,
        verbose=False,
    )["x"]

    return np.array(sample_weights_X).ravel()


class KKMM(KDR):
    """A K-KMM Density Ratio Biquality Classifier.

    A KDR Biquality Classifier using Ensemble [1]_
    Kernel Mean Matching [2]_ to reweigh untrusted examples [3]_.

    Parameters
    ----------
    estimator : object
        The estimator from which the KDR classifier is built.
        Support for sample weighting is required.

    kernel : str or callable, default="rbf"
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

    kernel_params : dict, optional (default={})
        Kernel additional parameters

    B: float, optional (default=1000)
        Bounding weights parameter.

    epsilon: float, optional (default=None)
        Constraint parameter.
        If ``None`` epsilon is set to
        ``(np.sqrt(n_samples_untrusted - 1)/np.sqrt(n_samples_untrusted)``.

    max_iter : int, default=100
        Maximum number of iterations. The solver iterates until convergence
        (determined by 'tol') or this number of iterations.

    tol: float, default=1e-4
        Termination criteria dictating the absolute and relative error
        on the primal residual, dual residual and duality gap.

    batch_size : int or float, default=None
        Size of minibatches for batched Kernel Mean Matching.
        An int value represent an absolute number of untrusted samples used per batch.
        An float value represent the fraction of untrusted samples used per batch.
        When set to None, use the entire untrusted samples in one batch.

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
    .. [1] Miao Y., Farahat A. and Kamel M.\
        "Ensemble Kernel Mean Matching", 2015

    .. [2] Huang, J. and Smola, A. and Gretton, A. and Borgwardt, KM.\
        and Schölkopf, B., "Correcting Sample Selection Bias by Unlabeled Data", 2006

    .. [3] Fang, T., Lu, N., Niu, G., and Sugiyama, M.\
        "Rethinking importance weighting for deep learning\
        under distribution shift.", NeurIPS 2020
    """

    def __init__(
        self,
        estimator,
        *,
        kernel="rbf",
        kernel_params={},
        B=1000,
        epsilon=None,
        max_iter=1000,
        tol=1e-6,
        batch_size=None,
        n_jobs=None,
    ):
        super().__init__(estimator=estimator, n_jobs=n_jobs)

        self.kernel = kernel
        self.kernel_params = kernel_params
        self.B = B
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        self.batch_size = batch_size

    def _density_ratio(self, X_untrusted_i, X_trusted_i):
        """Implement reweight by ratio density estimation
        using Ensemble KernelMeanMatching."""

        n_samples_untrusted_i = _num_samples(X_untrusted_i)

        if isinstance(self.batch_size, float):
            if not 0 < self.batch_size <= 1:
                raise ValueError("""When `batch_size` is provided as a float,
                    it needs to be between 0 (exclusive) and 1 (inclusive).""")
            batch_size = int(self.batch_size * n_samples_untrusted_i)
        elif isinstance(self.batch_size, int):
            if not 1 <= self.batch_size:
                raise ValueError("""When `batch_size` is provided as an int,
                    it needs to be superior or equal to 1.""")
            batch_size = self.batch_size
        elif self.batch_size is None:
            batch_size = n_samples_untrusted_i
        else:
            raise ValueError("""Unknown `batch_size` %s.""" % self.batch_size)

        if batch_size < 1 or batch_size > n_samples_untrusted_i:
            warnings.warn("""Computed `batch_size` is less than 1 or larger than
                the number of untrusted samples for this class.
                It is going to be clipped.""")
            batch_size = np.clip(batch_size, 1, n_samples_untrusted_i)

        batch_slices = gen_batches(n_samples_untrusted_i, batch_size)

        kmms = Parallel(n_jobs=self.n_jobs)(
            delayed(kmm)(
                X_untrusted_i[batch_slice],
                X_trusted_i,
                kernel=self.kernel,
                kernel_params=self.kernel_params,
                B=self.B,
                epsilon=self.epsilon,
                max_iter=self.max_iter,
                tol=self.tol,
                n_jobs=self.n_jobs,
            )
            for batch_slice in batch_slices
        )

        return np.concatenate(kmms)


class IKMM(IDR):
    """An Iterative KMM Density Ratio Biquality Classifier.

    An Iterative DR Biquality Classifier using Ensemble [1]_
    Kernel Mean Matching [2]_ to reweigh untrusted examples [3]_.

    Parameters
    ----------
    estimator : object
        The estimator from which the IDR classifier is built.
        Support for sample weighting and probability prediction is required.

    n_estimators : int, default=10
        Number of trained estimators on reweighted samples.

    exploit_iterative_learning: boolean, default=False
        If the `estimator` supports iterative learning with `warm_start`,
        exploit it by computing new weights for every epoch when fitting
        `estimator`.

    window: int, default=1
        Number of previous losses used to compute sample weights.

    kernel : str or callable, default="rbf"
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

    kernel_params : dict, optional (default={})
        Kernel additional parameters

    B: float, optional (default=1000)
        Bounding weights parameter.

    epsilon: float, optional (default=None)
        Constraint parameter.
        If ``None`` epsilon is set to
        ``(np.sqrt(n_samples_untrusted - 1)/np.sqrt(n_samples_untrusted)``.

    max_iter : int, default=100
        Maximum number of iterations. The solver iterates until convergence
        (determined by 'tol') or this number of iterations.

    tol: float, default=1e-4
        Termination criteria dictating the absolute and relative error
        on the primal residual, dual residual and duality gap.

    batch_size : int or float, default=None
        Size of minibatches for batched Kernel Mean Matching.
        An int value represent an absolute number of untrusted samples used per batch.
        An float value represent the fraction of untrusted samples used per batch.
        When set to None, use the entire untrusted samples in one batch.

    n_jobs : int, default=None
        The number of jobs to use for the computation. This parallelize the
        density ratio estimation procedures on all samples.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    estimator_ : classifier
        The fitted estimator.

    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    n_classes_ : int
        The number of classes.

    References
    ----------
    .. [1] Miao Y., Farahat A. and Kamel M.\
        "Ensemble Kernel Mean Matching", 2015

    .. [2] Huang, J. and Smola, A. and Gretton, A. and Borgwardt, KM.\
        and Schölkopf, B., "Correcting Sample Selection Bias by Unlabeled Data", 2006

    .. [3] Fang, T., Lu, N., Niu, G., and Sugiyama, M.\
        "Rethinking importance weighting for deep learning\
        under distribution shift.", NeurIPS 2020
    """

    def __init__(
        self,
        estimator,
        *,
        n_estimators=10,
        exploit_iterative_learning=True,
        window=1,
        kernel="rbf",
        kernel_params={},
        B=1000,
        epsilon=None,
        max_iter=1000,
        tol=1e-6,
        batch_size=None,
        n_jobs=None,
    ):
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            exploit_iterative_learning=exploit_iterative_learning,
            window=window,
        )

        self.kernel = kernel
        self.kernel_params = kernel_params
        self.B = B
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        self.batch_size = batch_size
        self.n_jobs = n_jobs

    def _density_ratio(self, loss_untrusted, loss_trusted):
        """Implement reweight by ratio density estimation
        using Ensemble KernelMeanMatching."""

        n_samples_untrusted = _num_samples(loss_untrusted)

        if isinstance(self.batch_size, float):
            if not 0 < self.batch_size <= 1:
                raise ValueError("""When `batch_size` is provided as a float,
                    it needs to be between 0 (exclusive) and 1 (inclusive).""")
            batch_size = int(self.batch_size * n_samples_untrusted)
        elif isinstance(self.batch_size, int):
            if not 1 <= self.batch_size:
                raise ValueError("""When `batch_size` is provided as an int,
                    it needs to be superior or equal to 1.""")
            batch_size = self.batch_size
        elif self.batch_size is None:
            batch_size = n_samples_untrusted
        else:
            raise ValueError("""Unknown `batch_size` %s.""" % self.batch_size)

        if batch_size < 1 or batch_size > n_samples_untrusted:
            warnings.warn("""Computed `batch_size` is less than 1 or larger than
                the number of untrusted samples for this class.
                It is going to be clipped.""")
            batch_size = np.clip(batch_size, 1, n_samples_untrusted)

        batch_slices = gen_batches(n_samples_untrusted, batch_size)

        kmms = Parallel(n_jobs=self.n_jobs)(
            delayed(kmm)(
                loss_untrusted[batch_slice],
                loss_trusted,
                kernel=self.kernel,
                kernel_params=self.kernel_params,
                B=self.B,
                epsilon=self.epsilon,
                max_iter=self.max_iter,
                tol=self.tol,
                n_jobs=self.n_jobs,
            )
            for batch_slice in batch_slices
        )

        return np.concatenate(kmms)
