import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold

from bqlearn.model_selection import make_biquality_cv


def test_mbqcv_n_splits_equals_cv_n_splits():
    X, _ = make_classification()
    sample_quality = np.random.randint(0, 2, size=(100,))

    cv = KFold()
    bqcv = make_biquality_cv(X, sample_quality, cv=cv)

    assert cv.get_n_splits() == bqcv.get_n_splits()


def test_mbqcv_no_untrusted_in_test():
    X, y = make_classification()
    sample_quality = np.random.randint(0, 2, size=(100,))

    bqcv = make_biquality_cv(X, sample_quality, cv=5)

    for _, test in bqcv.split(X, y):
        assert np.all(sample_quality[test] == 1)


def test_mbqcv_preserves_trusted_train_and_add_all_untrusted_in_every_train():
    X, y = make_classification()
    sample_quality = np.random.randint(0, 2, size=(100,))

    cv = KFold()
    bqcv = make_biquality_cv(X, sample_quality, cv=cv)

    for (train, _), (bqtrain, _) in zip(cv.split(X, y), bqcv.split(X, y)):
        assert np.all(np.isin(train, bqtrain))
