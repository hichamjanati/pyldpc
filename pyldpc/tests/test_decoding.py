import numpy as np

from pyldpc import (make_ldpc, binaryproduct, encode_random_message, decode,
                    get_message, encode)
import pytest
from itertools import product


@pytest.mark.parametrize("systematic, log, sparse",
                         product([False, True], [False, True], [False, True]))
def test_decoding_random(systematic, log, sparse):
    n = 15
    d_v = 4
    d_c = 5
    seed = 0

    H, G = make_ldpc(n, d_v, d_c, seed=seed, systematic=systematic,
                     sparse=sparse)
    assert not binaryproduct(H, G).any()
    n, k = G.shape
    snr = 100

    v, y = encode_random_message(G, snr, seed)

    d = decode(H, y, snr, maxiter=50, log=log)
    x = get_message(G, d)

    assert abs(v - x).sum() == 0


@pytest.mark.parametrize("systematic, log, sparse",
                         product([False, True], [True, False], [False, True]))
def test_decoding(systematic, log, sparse):
    n = 15
    d_v = 4
    d_c = 5
    seed = 0

    H, G = make_ldpc(n, d_v, d_c, seed=seed, systematic=systematic,
                     sparse=sparse)
    assert not binaryproduct(H, G).any()

    n, k = G.shape
    snr = 100

    v = np.arange(k) % 2
    y = encode(G, v, snr, seed)

    d = decode(H, y, snr, maxiter=10, log=log)
    x = get_message(G, d)

    assert abs(v - x).sum() == 0
