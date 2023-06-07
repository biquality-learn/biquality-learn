import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold

from bqlearn.model_selection import make_biquality_cv


def test_mbqcv_no_untrusted_in_test():
    X, y = make_classification()
    sample_quality = np.random.randint(0, 1, size=(100,))

    cv = make_biquality_cv(X, sample_quality, cv=5)

    for _, test in cv.split(X, y):
        assert np.all(sample_quality[test] == 0)


def test_mbqcv_do_not_modify_train():
    X, y = make_classification()
    sample_quality = np.random.randint(0, 1, size=(100,))

    cv = KFold()

    bqcv = make_biquality_cv(X, sample_quality, cv=cv)

    for (train, _), (bqtrain, _) in zip(cv.split(X, y), bqcv.split(X, y)):
        assert np.array_equal(train, bqtrain)
