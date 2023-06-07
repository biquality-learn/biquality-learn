import numpy as np
import pytest

from bqlearn.corruptions.noise_matrices import (
    background_noise_matrix,
    flip_noise_matrix,
    uniform_noise_matrix,
)


@pytest.mark.parametrize("noise_ratio", [0.1, 1.0])
@pytest.mark.parametrize("n_classes", [10, 100])
@pytest.mark.parametrize(
    "noise_matrix", [uniform_noise_matrix, flip_noise_matrix, background_noise_matrix]
)
def test_sum_rows_matrices_equals_1(noise_ratio, n_classes, noise_matrix):
    assert np.allclose(
        np.ones(n_classes), np.sum(noise_matrix(n_classes, noise_ratio), axis=1)
    )


@pytest.mark.parametrize("n_classes", [10, 100])
@pytest.mark.parametrize(
    "noise_matrix", [uniform_noise_matrix, flip_noise_matrix, background_noise_matrix]
)
def test_clean_noise_matrix_is_identity(n_classes, noise_matrix):
    assert np.allclose(np.identity(n_classes), noise_matrix(n_classes, 0.0))


@pytest.mark.parametrize("n_classes", [10, 100])
def test_noisy_uniform_noise_matrix(n_classes):
    assert np.all(uniform_noise_matrix(n_classes, 1.0) == 1 / n_classes)


@pytest.mark.parametrize("n_classes", [3, 10, 100])
@pytest.mark.parametrize("permutation", [True, False])
def test_noisy_flip_noise_matrix(n_classes, permutation):
    noise_matrix = flip_noise_matrix(n_classes, 1.0, permutation=permutation)

    assert np.allclose(np.ones(n_classes), np.sum(noise_matrix, axis=1))
    if permutation:
        assert np.allclose(np.ones(n_classes), np.sum(noise_matrix, axis=0))
    assert np.allclose(np.zeros(n_classes), np.diag(noise_matrix))
