"""
========================================
Coding - Decoding simulation of an image
========================================

This example shows a simulation of the transmission of an image as a
binary message through a gaussian white noise channel with an LDPC coding and
decoding system.
"""


import numpy as np
from pyldpc import (make_ldpc, binaryproduct, ldpc_images)
from pyldpc.utils_img import gray2bin, rgb2bin
from matplotlib import pyplot as plt
from PIL import Image

n = 69
d_v = 2
d_c = 3
seed = 42

##################################################################
# First we create an LDPC code i.e a pair of decoding and coding matrices
# H and G. H is a regular parity-check matrix with d_v ones per row
# and d_c ones per column
# The constatns d_c, d_c and n are chosen manually so that k = 24
# which corresponds to the number of bits of an RGB pixel

H, G = make_ldpc(n, d_v, d_c, seed=seed, systematic=True, sparse=True)

##################################################################
# Now we simulate transmission for different levels of noise and
# compute the percentage of errors using the bit-error-rate score
# and visualize the images

img = np.array(Image.open("data/eye.jpg"))
img_bin = rgb2bin(img)

errors = []
snrs = np.arange(9, 10, 2)
for snr in snrs:
    coded, noisy = ldpc_images.encode_img(G, img_bin, snr, seed)

    x = ldpc_images.decode_img(G, H, coded, snr, maxiter=100, log=False)
    print(x.shape, img.shape)
    # error = abs(img - x).sum() / k
    # errors.append(error)
#
# plt.figure()
# plt.plot(snrs, errors, color="indianred")
# plt.ylabel("Bit error rate")
# plt.xlabel("SNR")
# plt.show()
