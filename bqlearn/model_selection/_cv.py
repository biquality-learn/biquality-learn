import warnings

import numpy as np
from sklearn.model_selection import check_cv, PredefinedSplit
from sklearn.utils.validation import _check_sample_weight, _num_samples


def make_biquality_cv(X, sample_quality, cv=None, *, y=None, groups=None):
    """Utility function for building a biquality cross-validator.

    In the Biquality Data setup, cross-validators behave the same way as usual
    cross-validators, but untrusted samples should be remove from the generated
    test dataset.

    At the moment this cross-validator is made thanks to :class:`PredifinedSplit`
    and untrusted samples are removed from all test sets generated
    by the provided ``cv``. That's why each sample should be attributed
    to only one test set at maximum, otherwise a warning is returned.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The samples.

    sample_quality : array-like of shape (n_samples,)
        The sample quality.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold cross validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable that generates (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is either binary or multiclass,
        :class:`StratifiedKFold` is used. In all other cases,
        :class:`KFold` is used.

    y : array-like of shape (n_samples,), default=None
        The target variable.

    groups : array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset into
        train/test set.

    Returns
    -------
    biquality_cv : a cross-validator instance.
        The return value is a cross-validator which generates the train/test
        splits via the ``split`` method.
    """

    sample_quality = _check_sample_weight(sample_quality, X)

    cv = check_cv(cv, y, classifier=True)

    n_samples = _num_samples(X)

    folds = np.full(n_samples, -1.0)

    for i, (_, test_ind) in enumerate(cv.split(X, y, groups)):
        if np.any(folds[test_ind] != -1):
            warnings.warn(
                "Some samples appeared in multiple test sets, the last test found"
                " overrides the previously found test set for these samples.",
                UserWarning,
            )

        folds[test_ind] = i

    folds[sample_quality == 0] = -1

    return PredefinedSplit(folds)
