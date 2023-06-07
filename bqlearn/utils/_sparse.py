import numpy as np
import scipy.sparse as sp
from scipy.sparse import issparse


def safe_sparse_vstack(tup, dense_output=False):
    """Vertical stacking that handle the sparse matrix case correctly.

    Parameters
    ----------
    a : {ndarray, sparse matrix}

    b : {ndarray, sparse matrix}

    dense_output : bool, default=False
        When False, ``a`` and ``b`` both being sparse will yield sparse output.
        When True, output will always be a dense array.

    Returns
    -------
    vstacked : {ndarray, sparse matrix}
        Sparse if ``a`` and ``b`` are sparse and ``dense_output=False``.
    """
    if all([issparse(x) for x in tup]):
        ret = sp.vstack(tup)
    else:
        ret = np.vstack(tup)

    if all([issparse(x) for x in tup]) and dense_output and hasattr(ret, "toarray"):
        return ret.toarray()
    return ret
