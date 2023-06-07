import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from .. import kmm, pdr


def test_empty_dr():
    # Check that empty arguments yield ValueError
    with pytest.raises(ValueError):
        pdr(np.empty((0, 10)), np.random.rand(1, 10), LogisticRegression())
    with pytest.raises(ValueError):
        kmm(np.empty((0, 10)), np.random.rand(1, 10))
