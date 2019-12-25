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
    rng = utils.check_random_state(seed)
    n, k = tG.shape

    v = rng.randint(2, size=k)

    d = utils.binaryproduct(tG, v)
    x = (-1) ** d

    sigma = 10 ** (-snr / 20)

    e = rng.randn(n) * sigma

    y = x + e

    return v, y


def encode(tG, v, snr, seed=None):
    """Encode a binary message and adds Gaussian noise.

    Parameters
    ----------
    tG: array or scipy.sparse.csr_matrix (m, k). Transposed coding matrix
    obtained from `pyldpc.make_ldpc`.

    v: array (k, ) or (k, n_messages) binary messages to be encoded.

    snr: float. Signal-Noise Ratio. SNR = 10log(1 / variance) in decibels.

    Returns
    -------
    y: array (n,) or (n, n_messages) coded messages + noise.

    """
    n, k = tG.shape

    rng = utils.check_random_state(seed)
    d = utils.binaryproduct(tG, v)
    x = (-1) ** d

    sigma = 10 ** (- snr / 20)
    e = rng.randn(*x.shape) * sigma

    y = x + e

    return y
