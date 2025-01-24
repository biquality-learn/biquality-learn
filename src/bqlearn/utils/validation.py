from sklearn.metrics import confusion_matrix
from sklearn.utils import safe_mask

from bqlearn.metrics import (
    anchor_transition_matrix,
    gold_transition_matrix,
    iterative_anchor_transition_matrix,
)

__all__ = ["compute_transition_matrix"]


def compute_transition_matrix(
    tm, estimator, X, y, sample_quality, *, quantile=None, n_iter=None
):
    if tm == "anchor":
        transition_matrix = anchor_transition_matrix(
            estimator.predict_proba(X[safe_mask(X, sample_quality == 0)]),
            quantile=quantile,
        )
    elif tm == "iterative":
        transition_matrix = iterative_anchor_transition_matrix(
            estimator.predict_proba(X[safe_mask(X, sample_quality == 0)]),
            quantile=quantile,
            n_iter=n_iter,
        )
    elif tm == "gold":
        transition_matrix = gold_transition_matrix(
            y[sample_quality == 1],
            estimator.predict_proba(X[safe_mask(X, sample_quality == 1)]),
        )
    elif tm == "confusion":
        transition_matrix = confusion_matrix(
            y[sample_quality == 1],
            estimator.predict(X[safe_mask(X, sample_quality == 1)]),
            normalize="true",
        )
    else:
        raise ValueError(f"Unsupported transition matrix : {tm}")

    return transition_matrix
