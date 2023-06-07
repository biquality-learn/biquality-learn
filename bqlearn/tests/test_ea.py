import numpy as np
from sklearn.linear_model import LogisticRegression

from bqlearn.ea import EasyADAPT


def test_augment_easy_adapt():
    n_samples = 1000
    n_features = 10
    X = np.ones((n_samples, n_features))
    ea = EasyADAPT(LogisticRegression()).fit(X, np.random.randint(0, 2, size=1000))
    X_aug = ea.augment(X)

    assert X_aug.shape == (n_samples, 3 * n_features)
    assert np.array_equal(X_aug[:, 0:n_features], X)
    assert np.all(X_aug[:, n_features : 2 * n_features] == 0)
    assert np.array_equal(X_aug[:, 2 * n_features : 3 * n_features], X)
