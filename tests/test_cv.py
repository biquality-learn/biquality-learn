import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold

from bqlearn.model_selection import BiqualityCrossValidator


def test_bqcv_n_splits_equals_cv_n_splits():
    cv = KFold()
    bqcv = BiqualityCrossValidator(cv)

    assert cv.get_n_splits() == bqcv.get_n_splits()


def test_bqcv_default_cv_is_stratified():
    bqcv = BiqualityCrossValidator()

    assert isinstance(bqcv.cv, StratifiedKFold)


def test_bqcv_no_untrusted_in_test():
    X, y = make_classification()
    sample_quality = np.random.randint(0, 2, size=(100,))

    bqcv = BiqualityCrossValidator()

    for _, test in bqcv.split(X, y, groups=sample_quality):
        assert np.all(sample_quality[test] == 1)


def test_bqcv_all_untrusted_in_every_train():
    X, y = make_classification()
    sample_quality = np.random.randint(0, 2, size=(100,))

    untrusted = np.flatnonzero(sample_quality == 0)

    bqcv = BiqualityCrossValidator()

    for train, _ in bqcv.split(X, y, groups=sample_quality):
        assert np.all(np.isin(untrusted, train))


def test_bqcv_preserves_test():
    X, y = make_classification()
    sample_quality = np.random.randint(0, 2, size=(100,))

    trusted = np.flatnonzero(sample_quality == 1)

    cv = KFold()
    bqcv = BiqualityCrossValidator(cv)

    for (train, test), (bqtrain, bqtest) in zip(
        cv.split(X[sample_quality == 1], y[sample_quality == 1]),
        bqcv.split(X, y, groups=sample_quality),
    ):
        assert np.array_equal(trusted[test], bqtest)
        assert np.all(np.isin(trusted[train], bqtrain))


def test_bqcv_no_sample_quality_equals_cv():
    X, y = make_classification()

    cv = ShuffleSplit(random_state=1)
    bqcv = BiqualityCrossValidator(cv)

    for (train, test), (bqtrain, bqtest) in zip(
        cv.split(X, y),
        bqcv.split(X, y),
    ):
        assert np.array_equal(train, bqtrain)
        assert np.array_equal(test, bqtest)
