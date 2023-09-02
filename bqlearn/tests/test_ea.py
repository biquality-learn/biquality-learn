import numpy as np
import pytest
import scipy.sparse as sp

from bqlearn.ea import EasyADAPT


def test_augment_easy_adapt():
    n_samples = 1000
    n_features = 10
    X = np.ones((n_samples, n_features))
    X_aug = EasyADAPT().fit_transform(X)

    assert X_aug.shape == (n_samples, 3 * n_features)
    assert np.array_equal(X_aug[:, 0:n_features], X)
    assert np.all(X_aug[:, n_features : 2 * n_features] == 0)
    assert np.array_equal(X_aug[:, 2 * n_features : 3 * n_features], X)


@pytest.mark.parametrize("format", [sp.csr_matrix, sp.csc_matrix, sp.lil_matrix])
def test_easy_adapt_sparse(format):
    n_samples = 1_000_000
    n_features = 1_000
    X = format((n_samples, n_features))
    X_aug = EasyADAPT().fit_transform(X)

    assert X_aug.shape == (n_samples, 3 * n_features)
    assert X_aug.format == "csr"
