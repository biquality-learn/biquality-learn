"""
The :mod:`bqlearn.corruption` module implements a variety of synthetic corruption to
artificially make biquality datasets.
"""

from ._imbalance import make_cluster_imbalance, make_imbalance
from ._label_noise import (
    make_feature_dependent_label_noise,
    make_instance_dependent_label_noise,
    make_label_noise,
    noisy_leaves_probability,
    uncertainty_noise_probability,
)
from ._sampling_biais import make_sampling_biais
from ._weak_labels import make_weak_labels

__all__ = [
    "make_weak_labels",
    "make_label_noise",
    "make_instance_dependent_label_noise",
    "uncertainty_noise_probability",
    "noisy_leaves_probability",
    "make_imbalance",
    "make_cluster_imbalance",
    "make_feature_dependent_label_noise",
    "make_sampling_biais",
]
