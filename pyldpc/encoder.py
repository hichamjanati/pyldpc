import numpy as np
from . import utils


def encode_random_message(tG, snr, seed=None):
    """Encode a random message given a generating matrix tG and a SNR.

    Parameters
    ----------
    tG: array or scipy.sparse.csr_matrix (m, k). Transposed coding matrix
    obtained from `pyldpc.make_ldpc`.
    snr: float. Signal-Noise Ratio. SNR = 10log(1 / variance) in decibels.

    Returns
    -------
    v: array (k,) random message generated.
    y: array (n,) coded message + noise.

    """
    rnd = np.random.RandomState(seed)
    n, k = tG.shape

    v = rnd.randint(2, size=k)

    d = utils.binaryproduct(tG, v)
    x = (-1) ** d

    sigma = 10 ** (-snr / 20)

    e = rnd.randn(n) * sigma

    y = x + e

    return v, y


def encode(tG, v, snr, seed=None):
    """Encode a binary message and adds Gaussian noise.

    Parameters
    ----------
    tG: array or scipy.sparse.csr_matrix (m, k). Transposed coding matrix
    obtained from `pyldpc.make_ldpc`.

    v: array (k,) binary message to be encoded.

    snr: float. Signal-Noise Ratio. SNR = 10log(1 / variance) in decibels.

    Returns
    -------
    y: array (n,) coded message + noise.

    """
    n, k = tG.shape

    rnd = np.random.RandomState(seed)
    d = utils.binaryproduct(tG, v)
    x = (-1) ** d

    sigma = 10 ** (- snr / 20)
    e = rnd.randn(n) * sigma

    y = x + e

    return y
