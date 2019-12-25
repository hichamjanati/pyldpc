import numpy as np

from pyldpc import (make_ldpc, binaryproduct, ldpc_audio)
from pyldpc.utils_audio import audio2bin
import pytest
from itertools import product


@pytest.mark.parametrize("systematic, sparse",
                         product([True, False], [False, True]))
def test_audio(systematic, sparse):

    n = 48
    d_v = 2
    d_c = 3
    seed = 0
    rnd = np.random.RandomState(seed)
    H, G = make_ldpc(n, d_v, d_c, seed=seed, systematic=systematic,
                     sparse=sparse)
    assert not binaryproduct(H, G).any()

    n, k = G.shape
    snr = 1000

    audio = rnd.randint(0, 255, size=2)
    audio_bin = audio2bin(audio)
    coded, noisy = ldpc_audio.encode_audio(G, audio_bin, snr, seed)
    x = ldpc_audio.decode_audio(G, H, coded, snr, audio_bin.shape)
    ber = ldpc_audio.ber_audio(audio_bin, audio2bin(x))
    assert ber == 0
