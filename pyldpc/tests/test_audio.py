import numpy as np
import scipy

from pyldpc import (make_ldpc, binaryproduct, ldpc_audio)
from pyldpc.utils_audio import audio2bin
import pytest
from itertools import product


sparse = True
log = True
systematic = True


@pytest.mark.parametrize("systematic, log, sparse",
                         product([True], [True, False], [False, True]))
def test_audio(systematic, log, sparse):

    n = 25
    d_v = 4
    d_c = 5
    seed = 0
    rnd = np.random.RandomState(seed)
    H, G = make_ldpc(n, d_v, d_c, seed=seed, systematic=systematic)
    assert not binaryproduct(H, G).any()

    n, k = G.shape
    print(k)
    snr = 100
    if sparse:
        G = scipy.sparse.csr_matrix(G)
        H = scipy.sparse.csr_matrix(H)

    audio = rnd.randint(0, 255, size=5)
    audio_bin = audio2bin(audio)
    coded, noisy = ldpc_audio.encode_audio(G, audio_bin, snr, seed)

    x = ldpc_audio.decode_audio(G, H, coded, snr, maxiter=100, log=log)

    assert abs(audio - x).sum() == 0
