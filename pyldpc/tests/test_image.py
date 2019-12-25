import numpy as np

from pyldpc import (make_ldpc, binaryproduct, ldpc_images)
from pyldpc.utils_img import gray2bin, rgb2bin
import pytest
from itertools import product


@pytest.mark.parametrize("systematic, sparse",
                         product([False, True], [False, True]))
def test_image_gray(systematic, sparse):
    n = 100
    d_v = 3
    d_c = 4
    seed = 0
    rnd = np.random.RandomState(seed)
    H, G = make_ldpc(n, d_v, d_c, seed=seed, systematic=systematic,
                     sparse=sparse)
    assert not binaryproduct(H, G).any()

    n, k = G.shape
    snr = 1e3

    img = rnd.randint(0, 255, size=(3, 3))
    img_bin = gray2bin(img)
    img_shape = img_bin.shape

    coded, noisy = ldpc_images.encode_img(G, img_bin, snr, seed)

    x = ldpc_images.decode_img(G, H, coded, snr, img_shape=img_shape)

    assert ldpc_images.ber_img(img_bin, gray2bin(x)) == 0


@pytest.mark.parametrize("systematic, sparse",
                         product([False, True], [False, True]))
def test_image_rgb(systematic, sparse):

    n = 69
    d_v = 2
    d_c = 3
    seed = 0
    rnd = np.random.RandomState(seed)
    H, G = make_ldpc(n, d_v, d_c, seed=seed, systematic=systematic,
                     sparse=sparse)
    assert not binaryproduct(H, G).any()

    n, k = G.shape
    snr = 1e3

    img = rnd.randint(0, 255, size=(3, 3, 3))
    img_bin = rgb2bin(img)
    coded, noisy = ldpc_images.encode_img(G, img_bin, snr, seed)

    x = ldpc_images.decode_img(G, H, coded, snr, img_bin.shape)

    assert ldpc_images.ber_img(img_bin, rgb2bin(x)) == 0
