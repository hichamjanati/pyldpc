import numpy as np
from . import utils


def encode_random_message(tG, snr, seed=None):
    """
    IMPORTANT: tG can be transposed coding matrix scipy.sparse.csr_matrix
    object to speed up calculations. Randomly computes a k-bits message v,
    where G's shape is (k,n). And sends it through the canal.

    Message v is passed to G: d = v,G. d is a n-vector turned into a BPSK
    modulated vector x. Then Additive White Gaussian Noise (AWGN) is added.

    SNR is the Signal-Noise Ratio: SNR = 10log(1/variance) in decibels, where
    variance is the variance of the AWGN.
    Remember:
        1. d = v.G => tG.tv = td
        2. x = BPSK(d) (or if you prefer the math: x = pow(-1,d) )
        3. y = x + AWGN(0, snr)

    ----------------------------

    Parameters:

    tG: 2D-Array (OR scipy.sparse.csr_matrix) Transpose of Coding Matrix
    obtained from CodingMatrix functions.

    snr: the Signal-Noise Ratio: SNR = 10log(1 / variance) in decibels.
        >> snr = 10log(1 / variance)

    -------------------------------

    Returns

    Tuple(v,y) (Returns v to keep track of the random message v)

    """
    rnd = np.random.RandomState(seed)
    n, k = tG.shape

    v = rnd.randint(2, size=k)

    d = utils.binaryproduct(tG, v)
    x = pow(-1, d)

    sigma = 10 ** (-snr / 20)

    e = rnd.randn(n) * sigma

    y = x + e

    return v, y


def encode(tG, v, snr, seed=None):
    """

    IMPORTANT: if H is large, tG can be transposed coding matrix
    scipy.sparse.csr_matrix object to speed up calculations.
    Codes a message v with Coding Matrix G, and sends it through a
    noisy (default) channel. Message v is passed to tG: d = tG.tv d is a
    n-vector turned into a BPSK modulated vector x. Then Additive White
    Gaussian Noise (AWGN) is added.
    SNR is the Signal-Noise Ratio: SNR = 10log(1/variance) in decibels,
    where variance is the variance of the AWGN.

        1. d = v.G (or (td = tG.tv))
        2. x = BPSK(d) (or if you prefer the math: x = pow(-1,d) )
        3. y = x + AWGN(0,snr)

    Parameters:

    tG: 2D-Array (OR scipy.sparse.csr_matrix)Transposed Coding Matrix obtained
    from coding_matrix functions. v: 1D-Array, k-vector (binary of course ..)
    SNR: Signal-Noise-Ratio: SNR = 10log(1/variance) in decibels.

    -------------------------------

    Returns y

    """
    n, k = tG.shape

    rnd = np.random.RandomState(seed)
    d = utils.binaryproduct(tG, v)
    x = (-1) ** d

    sigma = 10 ** (- snr / 20)
    e = rnd.randn(n) * sigma

    y = x + e

    return y
