"""
================================================
Coding - Decoding simulation of a random message
================================================

This example shows a simulation of the transmission of a binary message
through a gaussian white noise channel with an LDPC coding and decoding system.
"""


import numpy as np
from pyldpc import (make_ldpc, binaryproduct, decode, get_message, encode)
from matplotlib import pyplot as plt

n = 100
d_v = 4
d_c = 5
seed = 42

##################################################################
# First we create an LDPC code i.e a pair of decoding and coding matrices
# H and G. H is a regular parity-check matrix with d_v ones per row
# and d_c ones per column

H, G = make_ldpc(n, d_v, d_c, seed=seed, systematic=True, sparse=True)
# test if H and G are orthogonal
assert not binaryproduct(H, G).any()
n, k = G.shape
print("Code length:", k)

##################################################################
# Now we simulate transmission for different levels of noise and
# compute the percentage of errors using the bit-error-rate score


errors = []
snrs = np.arange(-10, 10, 2)
v = np.arange(k) % 2  # fixed k bits message
for snr in snrs:
    y = encode(G, v, snr, seed)
    d = decode(H, y, snr, maxiter=100, log=True)
    x = get_message(G, d)
    error = abs(v - x).sum() / k
    errors.append(error)

plt.figure()
plt.plot(snrs, errors, color="indianred")
plt.ylabel("Bit error rate")
plt.xlabel("SNR")
plt.show()
