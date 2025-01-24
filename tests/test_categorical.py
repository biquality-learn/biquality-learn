import numpy as np
from sklearn.preprocessing import normalize

from bqlearn.utils import categorical


def test_right_shape():
    rng = np.random.RandomState(1)

    X = normalize(rng.rand(10, 4), norm="l1")
    y = categorical(X)

    assert y.shape[0] == X.shape[0]
    assert len(y.shape) == 1


def test_categorical_output_max_is_less_than_input_shape_1():
    rng = np.random.RandomState(1)

    X = normalize(rng.rand(10, 4), norm="l1")
    y = categorical(X)

    assert np.max(y) <= X.shape[1]
