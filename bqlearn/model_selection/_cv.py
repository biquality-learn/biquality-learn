import numbers

import numpy as np
from sklearn.model_selection import check_cv, StratifiedKFold
from sklearn.utils.validation import _num_samples, indexable


class BiqualityCrossValidator:
    """Biquality cross-validator.

    In the Biquality Data setup, cross-validators split only trusted data.
    All untrusted samples are present in each training splits and test sets contain
    only trusted samples.

    The `sample_quality` is provided through the `groups` argument
    of the :meth:`split` method.

    Parameters
    ----------

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold cross validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable that generates (train, test) splits as arrays of indices.

        For integer/None inputs, :class:`StratifiedKFold` is used.
    """

    def __init__(self, cv=None):
        cv = 5 if cv is None else cv
        if isinstance(cv, numbers.Integral):
            cv = StratifiedKFold(cv)
        self.cv = check_cv(cv)

    def split(self, X, y, groups=None):
        X, y, groups = indexable(X, y, groups)

        if groups is None:
            groups = np.ones(_num_samples(X))

        trusted = np.flatnonzero(groups == 1)
        untrusted = np.flatnonzero(groups == 0)

        mask = groups == 1

        for train, test in self.cv.split(X[mask], y[mask], groups[mask]):
            yield np.concatenate((trusted[train], untrusted)), trusted[test]

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.cv.get_n_splits(X, y, groups)
