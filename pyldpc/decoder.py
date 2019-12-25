"""Decoding module."""
import numpy as np
import warnings
from . import utils

from numba import njit, int64, types, float64


def decode(H, y, snr, maxiter=1000):
    """Decode a Gaussian noise corrupted n bits message using BP algorithm.

    Parameters
    ----------
    H: array (n_equations, n_code). Decoding matrix H.
    y: array (n_code,). Received message in the codeword space.
    maxiter: int. Maximum number of iterations of the BP algorithm.

    Returns
    -------
    x: array (n_code) the solution in the codeword space.

    """

    return _decode_logbp(H, y, snr, maxiter=maxiter)


def _decode_logbp(H, y, snr, maxiter=1000):
    """Perform BP algorithm"""
    m, n = H.shape

    var = 10 ** (-snr / 10)

    # step 0: initialisation

    Lc = 2 * y / var

    Lq = np.zeros(shape=(m, n), dtype=float)

    Lr = np.zeros(shape=(m, n), dtype=float)

    bits, nodes = utils.bitsandnodes(H)
    bits = np.array(bits)
    nodes = np.array(nodes)

    for n_iter in range(maxiter):
        Lq, Lr, L_posteriori = _inner_logbp_loop(bits, nodes, Lc, Lq, Lr,
                                                 n_iter)
        x = np.array(L_posteriori <= 0).astype(int)

        product = utils.incode(H, x)

        if product:
            break
    if n_iter == maxiter - 1:
        warnings.warn("""Decoding stopped before convergence. You may want
                       to increase maxiter""")
    return x


output_type_log = types.Tuple((float64[:, :], float64[:, :], float64[:]))


@njit(output_type_log(int64[:, :], int64[:, :], float64[:], float64[:, :],
                      float64[:, :], int64), cache=True)
def _inner_logbp_loop(bits, nodes, Lc, Lq, Lr, n_iter):
    """Perform insider Logbp loop in numba."""
    m, n = Lq.shape
    # setp 1 : Horizontal
    for i in range(m):
        ni = bits[i]
        n_bits_ = len(ni)
        for j_index in range(n_bits_):
            j = ni[j_index]
            nij = ni[:]
            X = 1.
            if n_iter == 0:
                for kk in range(len(nij)):
                    if kk != j:
                        X *= np.tanh(0.5 * Lc[nij[kk]])
            else:
                for kk in range(len(nij)):
                    if kk != j:
                        X *= np.tanh(0.5 * Lq[i, nij[kk]])

            num = 1 + X
            denom = 1 - X
            if num == 0:
                Lr[i, j] = -1
            elif denom == 0:
                Lr[i, j] = 1
            else:
                Lr[i, j] = np.log(num / denom)

    # step 2 : Vertical
    for j in range(n):
        mj = nodes[j]

        for i in mj:
            mji = mj[:]
            Lq[i, j] = Lc[j]
            for kk in range(len(mji)):
                if kk != i:
                    Lq[i, j] += Lr[mji[kk], j]

    # LLR a posteriori:
    L_posteriori = np.zeros(n)
    for j in range(n):
        mj = nodes[j]
        L_posteriori[j] = Lc[j] + Lr[mj, j].sum()

    return Lq, Lr, L_posteriori


def _decode_logbp_ext(H, bits, nodes, y, snr, maxiter=1000):
    """Perform BP algorithm in log-domain on specific nodes and bits."""
    m, n = H.shape

    var = 10 ** (-snr / 10)

    # step 0: initialization

    Lc = 2 * y / var

    Lq = np.zeros(shape=(m, n))

    Lr = np.zeros(shape=(m, n))

    for n_iter in range(maxiter):
        Lq, Lr, L_posteriori = _inner_ext_logbp(bits, nodes, Lc, Lq, Lr,
                                                n_iter)
        x = np.array(L_posteriori <= 0).astype(int)

        product = utils.incode(H, x)

        if product:
            break

    if n_iter == maxiter - 1:
        warnings.warn("""Decoding stopped before convergence. You may want
                       to increase maxiter""")
    return x


@njit(output_type_log(int64[:, :], int64[:, :], float64[:], float64[:, :],
                      float64[:, :], int64), cache=True)
def _inner_ext_logbp(bits, nodes, Lc, Lq, Lr, n_iter):
    """Perform inner ext LogBP solver."""
    m, n = Lr.shape
    # step 1 : Horizontal
    for i in range(m):
        ni = bits[i]
        for j in ni:
            nij = ni[:]

            X = 1.
            if n_iter == 0:
                for kk in range(len(nij)):
                    if kk != j:
                        X *= np.tanh(0.5 * Lc[nij[kk]])
            else:
                for kk in range(len(nij)):
                    if kk != j:
                        X *= np.tanh(0.5 * Lq[i, nij[kk]])
            num = 1 + X
            denom = 1 - X
            if num == 0:
                Lr[i, j] = -1
            elif denom == 0:
                Lr[i, j] = 1
            else:
                Lr[i, j] = np.log(num / denom)

    # step 2 : Vertical
    for j in range(n):
        mj = nodes[j]

        for i in mj:
            mji = mj[:]
            Lq[i, j] = Lc[j]

            for kk in range(len(mji)):
                if kk != i:
                    Lq[i, j] += Lr[mji[kk], j]

    # LLR a posteriori:
    L_posteriori = np.zeros(n)
    for j in range(n):
        mj = nodes[j]

        L_posteriori[j] = Lc[j] + Lr[mj, j].sum()

    return Lq, Lr, L_posteriori


def get_message(tG, x):
    """Compute the original `n_bits` message from a `n_code` codeword `x`.

    Parameters
    ----------
    tG: array (n_code, n_bits) coding matrix tG.
    x: array (n_code,) decoded codeword of length `n_code`.

    Returns
    -------
    message: array (n_bits,). Original binary message.

    """
    n, k = tG.shape

    rtG, rx = utils.gausselimination(tG, x)

    message = np.zeros(k).astype(int)

    message[k - 1] = rx[k - 1]
    for i in reversed(range(k - 1)):
        message[i] = rx[i]
        message[i] -= utils.binaryproduct(rtG[i, list(range(i+1, k))],
                                          message[list(range(i+1, k))])

    return abs(message)
