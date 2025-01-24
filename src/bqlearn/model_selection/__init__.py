"""
The :mod:`bqlearn.model_selection` module implements utility functions for selecting
the best possible models on biquality data.
"""

from ._cv import BiqualityCrossValidator

__all__ = ["BiqualityCrossValidator"]
