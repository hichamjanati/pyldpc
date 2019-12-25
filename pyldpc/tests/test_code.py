import numpy as np

from pyldpc.code import make_ldpc
import pytest


@pytest.mark.parametrize("n, d_v, d_c, systematic",
                         [[30, 2, 5, False], [20, 2, 4, False],
                          [10, 3, 5, True], [25, 2, 5, False]])
def test_ldpc_matrix(n, d_v, d_c, systematic):
    H, G = make_ldpc(n, d_v, d_c, systematic=systematic)
    np.testing.assert_equal(H.sum(0), d_v)
    np.testing.assert_equal(H.sum(1), d_c)


@pytest.mark.parametrize("systematic", [False, True])
def test_ldpc_matrix_errors(systematic):
    n = 10
    d_v, d_c = 1, 4
    with pytest.raises(ValueError, match="d_v must be at least 2"):
        H, G = make_ldpc(n, d_v, d_c, systematic=systematic)

    d_v, d_c = 4, 4
    with pytest.raises(ValueError, match="d_c must be greater than d_v"):
        H, G = make_ldpc(n, d_v, d_c, systematic=systematic)

    d_v, d_c = 2, 4
    with pytest.raises(ValueError, match="d_c must divide n"):
        H, G = make_ldpc(n, d_v, d_c, systematic=systematic)
