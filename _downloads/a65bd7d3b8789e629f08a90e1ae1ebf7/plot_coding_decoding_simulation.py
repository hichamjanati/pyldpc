"""
================================================
Coding - Decoding simulation of a random message
================================================

This example shows a simulation of the transmission of a binary message
through a gaussian white noise channel with an LDPC coding and decoding system.
"""


import numpy as np
from pyldpc import make_ldpc, decode, get_message, encode
from matplotlib import pyplot as plt

n = 30
d_v = 2
d_c = 3
seed = np.random.RandomState(42)
##################################################################
# First we create an LDPC code i.e a pair of decoding and coding matrices
# H and G. H is a regular parity-check matrix with d_v ones per row
# and d_c ones per column

H, G = make_ldpc(n, d_v, d_c, seed=seed, systematic=True, sparse=True)

n, k = G.shape
print("Number of coded bits:", k)

##################################################################
# Now we simulate transmission for different levels of noise and
# compute the percentage of errors using the bit-error-rate score


errors = []
snrs = np.linspace(-2, 10, 20)
v = np.arange(k) % 2  # fixed k bits message
n_trials = 50  # number of transmissions with different noise
for snr in snrs:
    error = 0.
    for ii in range(n_trials):
        y = encode(G, v, snr, seed=seed)
        d = decode(H, y, snr)
        x = get_message(G, d)
        error += abs(v - x).sum() / k
    errors.append(error / n_trials)

plt.figure()
plt.plot(snrs, errors, color="indianred")
plt.ylabel("Bit error rate")
plt.xlabel("SNR")
plt.show()
