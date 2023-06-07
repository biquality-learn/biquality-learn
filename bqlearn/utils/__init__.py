"""
The :mod:`bqlearn.utils` module provide utilities for biquality learning.
"""

from ._categorical import categorical
from ._sparse import safe_sparse_vstack

__all__ = [
    "categorical",
    "safe_sparse_vstack",
]
