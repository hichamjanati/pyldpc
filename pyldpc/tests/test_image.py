import numpy as np

from pyldpc import (make_ldpc, binaryproduct, ldpc_images)
from pyldpc.utils_img import gray2bin, rgb2bin
import pytest
from itertools import product


@pytest.mark.parametrize("systematic, log, sparse",
                         product([False, True], [True], [False, True]))
def test_image_gray(systematic, log, sparse):

    n = 21
    d_v = 2
    d_c = 3
    seed = 0
    rnd = np.random.RandomState(seed)
    H, G = make_ldpc(n, d_v, d_c, seed=seed, systematic=systematic,
                     sparse=sparse)
    assert not binaryproduct(H, G).any()

    n, k = G.shape
    snr = 100

    img = rnd.randint(0, 255, size=(3, 3))
    img_bin = gray2bin(img)

    coded, noisy = ldpc_images.encode_img(G, img_bin, snr, seed)

    x = ldpc_images.decode_img(G, H, coded, snr, maxiter=100, log=log)

    assert ldpc_images.ber_img(img_bin, gray2bin(x)) == 0


@pytest.mark.parametrize("systematic, log, sparse",
                         product([False, True], [True], [False, True]))
def test_image_rgb(systematic, log, sparse):

    n = 69
    d_v = 2
    d_c = 3
    seed = 0
    rnd = np.random.RandomState(seed)
    H, G = make_ldpc(n, d_v, d_c, seed=seed, systematic=systematic,
                     sparse=sparse)
    assert not binaryproduct(H, G).any()

    n, k = G.shape
    snr = 100

    img = rnd.randint(0, 255, size=(3, 3, 3))
    img_bin = rgb2bin(img)
    coded, noisy = ldpc_images.encode_img(G, img_bin, snr, seed)

    x = ldpc_images.decode_img(G, H, coded, snr, maxiter=100, log=log)

    assert ldpc_images.ber_img(img_bin, rgb2bin(x)) == 0


@pytest.mark.parametrize("systematic, log, sparse",
                         product([True], [True, False], [False, True]))
def test_image_row(systematic, log, sparse):
    n = 213
    d_v = 2
    d_c = 3
    seed = 0
    rnd = np.random.RandomState(seed)
    H, G = make_ldpc(n, d_v, d_c, seed=seed, systematic=systematic,
                     sparse=sparse)
    assert not binaryproduct(H, G).any()

    n, k = G.shape
    snr = 100

    img = rnd.randint(0, 255, size=(3, 3, 3))
    img_bin = rgb2bin(img)
    coded, noisy = ldpc_images.encode_img_rowbyrow(G, img_bin, snr, seed)

    x = ldpc_images.decode_img_rowbyrow(G, H, coded, snr, maxiter=100, log=log)

    assert ldpc_images.ber_img(img_bin, rgb2bin(x)) == 0
