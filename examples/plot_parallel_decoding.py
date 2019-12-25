"""
==================================
Parallelization of Log-BP decoding
==================================

This example shows how coding and decoding can be done in parallel to speed
up computations.
"""


import numpy as np
from pyldpc import make_ldpc, decode, get_message, encode
from matplotlib import pyplot as plt

n = 30
d_v = 2
d_c = 3
seed = 42
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
# To parallelize coding and decoding, simply stack the messages as columns:

errors = []
snrs = np.linspace(-2, 10, 20)
v = np.arange(k) % 2  # fixed k bits message
n_trials = 50  # number of transmissions with different noise
V = np.tile(v, (n_trials, 1)).T  # stack v in columns

for snr in snrs:
    y = encode(G, V, snr, seed=seed)
    D = decode(H, y, snr)
    error = 0.
    for i in range(n_trials):
        x = get_message(G, D[:, i])
        error += abs(v - x).sum() / (k * n_trials)
    errors.append(error)

plt.figure()
plt.plot(snrs, errors, color="indianred")
plt.ylabel("Bit error rate")
plt.xlabel("SNR")
plt.show()
