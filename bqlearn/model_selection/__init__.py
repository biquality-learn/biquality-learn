"""
The :mod:`bqlearn.model_selection` module implements utility functions for selecting
the best possible models on biquality data.
"""

from ._cv import make_biquality_cv

__all__ = ["make_biquality_cv"]
