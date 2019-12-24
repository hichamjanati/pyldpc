import numpy as np
import warnings
from . import utils


def decode(H, y, snr, maxiter=100, log=True):
    """Decodes a Gaussian noise corrupted n bits message using BP algorithm.

    Parameters
    ----------

    H: array (n_equations, n_code). """
    f = _decode_bp
    if log:
        f = _decode_logbp
    return f(H, y, snr, maxiter=maxiter)


def _decode_bp(H, y, snr, maxiter=100):
    """Decodes a Gaussian noise corrupted n bits message using BP algorithm."""

    m, n = H.shape
    sigma = 10 ** (-snr / 20)

    p0 = np.zeros(shape=n)
    p0 = utils.f1(y, sigma) / (utils.f1(y, sigma) + utils.fm1(y, sigma))
    p1 = np.zeros(shape=n)
    p1 = utils.fm1(y, sigma) / (utils.f1(y, sigma) + utils.fm1(y, sigma))

    # step 0 : Initialization
    q0 = np.zeros(shape=(m, n))
    q0[:] = p0

    q1 = np.zeros(shape=(m, n))
    q1[:] = p1

    r0 = np.zeros(shape=(m, n))
    r1 = np.zeros(shape=(m, n))

    count = 0

    bits, nodes = utils.bitsandnodes(H)

    while(True):
        # step 1: horizontal

        deltaq = q0 - q1
        deltar = r0 - r1

        for i in range(m):
            ni = bits[i]
            for j in ni:
                nij = ni[:]

                if j in nij:
                    nij.remove(j)
                deltar[i, j] = np.prod(deltaq[i, nij])

        r0 = 0.5 * (1 + deltar)
        r1 = 0.5 * (1 - deltar)

        # step 2: vertical

        for j in range(n):
            mj = nodes[j]
            for i in mj:
                mji = mj[:]
                if i in mji:
                    mji.remove(i)

                q0[i, j] = p0[j] * np.prod(r0[mji, j])
                q1[i, j] = p1[j] * np.prod(r1[mji, j])

                if q0[i, j] + q1[i, j] == 0:
                    q0[i, j] = 0.5
                    q1[i, j] = 0.5

                else:
                    alpha = 1 / (q0[i, j]+q1[i, j])
                    # normalization cosntant alpha[i, j]

                    q0[i, j] *= alpha
                    q1[i, j] *= alpha
                    # now q0[i, j] + q1[i, j] = 1

        # computing posterior probabilites:
        q0_post = np.zeros(n)
        q1_post = np.zeros(n)

        for j in range(n):
            mj = nodes[j]
            q0_post[j] = p0[j] * np.prod(r0[mj, j])
            q1_post[j] = p1[j] * np.prod(r1[mj, j])

            if q0_post[j] + q1_post[j] == 0:
                q0_post[j] = 0.5
                q1_post[j] = 0.5

            alpha = 1 / (q0_post[j] + q1_post[j])

            q0_post[j] *= alpha
            q1_post[j] *= alpha

        x = np.array(q1_post > q0_post).astype(int)

        if utils.incode(H, x) or count >= maxiter:
            break

    if count == maxiter:
        warnings.warn("""Decoding stopped before convergence. You may want
                       to increase maxiter""")
    return x


def _decode_logbp(H, y, snr, maxiter=100):
    """Performs BP algorithm in log-domain."""

    m, n = H.shape

    var = 10 ** (-snr / 10)

    # step 0: initialisation

    Lc = 2 * y / var

    Lq = np.zeros(shape=(m, n))

    Lr = np.zeros(shape=(m, n))

    count = 0

    bits, nodes = utils.bitsandnodes(H)

    while(True):

        count += 1

        # setp 1 : Horizontal
        for i in range(m):
            ni = bits[i]
            for j in ni:
                nij = ni[:]

                if j in nij:
                    nij.remove(j)
                if count == 1:
                    X = np.prod(np.tanh(0.5 * Lc[nij]))
                else:
                    X = np.prod(np.tanh(0.5 * Lq[i, nij]))
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
                if i in mji:
                    mji.remove(i)
                Lq[i, j] = Lc[j] + sum(Lr[mji, j])

        # LLR a posteriori:
        L_posteriori = np.zeros(n)
        for j in range(n):
            mj = nodes[j]
            L_posteriori[j] = Lc[j] + sum(Lr[mj, j])

        x = np.array(L_posteriori <= 0).astype(int)

        product = utils.incode(H, x)

        if product or count >= maxiter:
            break
    if count == maxiter:
        warnings.warn("""Decoding stopped before convergence. You may want
                       to increase maxiter""")
    return x


def _decode_bp_ext(H, bits, nodes, y, snr, maxiter=100):
    """BP algorithm on specific nodes and bits."""
    m, n = H.shape

    sigma = 10 ** (-snr / 20)

    p0 = np.zeros(shape=n)
    p0 = utils.f1(y, sigma) / (utils.f1(y, sigma) + utils.fm1(y, sigma))
    p1 = np.zeros(shape=n)
    p1 = utils.fm1(y, sigma) / (utils.f1(y, sigma) + utils.fm1(y, sigma))

    # step 0 : Initialization
    q0 = np.zeros(shape=(m, n))
    q0[:] = p0

    q1 = np.zeros(shape=(m, n))
    q1[:] = p1

    r0 = np.zeros(shape=(m, n))
    r1 = np.zeros(shape=(m, n))

    count = 0

    while(True):

        count += 1
        # step 1 : Horizontal

        deltaq = q0 - q1
        deltar = r0 - r1

        for i in range(m):
            ni = bits[i]
            for j in ni:
                nij = ni[:]

                if j in nij:
                    nij.remove(j)
                deltar[i, j] = np.prod(deltaq[i, nij])

        r0 = 0.5 * (1 + deltar)
        r1 = 0.5 * (1 - deltar)

        # step 2 : Vertical

        for j in range(n):
            mj = nodes[j]
            for i in mj:
                mji = mj[:]
                if i in mji:
                    mji.remove(i)

                q0[i, j] = p0[j] * np.prod(r0[mji, j])
                q1[i, j] = p1[j] * np.prod(r1[mji, j])

                if q0[i, j] + q1[i, j] == 0:
                    q0[i, j] = 0.5
                    q1[i, j] = 0.5

                else:
                    alpha = 1 / (q0[i, j] + q1[i, j])

                    q0[i, j] *= alpha
                    q1[i, j] *= alpha

        q0_post = np.zeros(n)
        q1_post = np.zeros(n)

        for j in range(n):
            mj = nodes[j]
            q0_post[j] = p0[j] * np.prod(r0[mj, j])
            q1_post[j] = p1[j] * np.prod(r1[mj, j])

            if q0_post[j] + q1_post[j] == 0:
                q0_post[j] = 0.5
                q1_post[j] = 0.5

            alpha = 1 / (q0_post[j] + q1_post[j])

            q0_post[j] *= alpha
            q1_post[j] *= alpha

        x = np.array(q1_post > q0_post).astype(int)

        if utils.incode(H, x) or count >= maxiter:
            break
    if count == maxiter:
        warnings.warn("""Decoding stopped before convergence. You may want
                       to increase maxiter""")
    return x


def _decode_logbp_ext(H, bits, nodes, y, snr, maxiter=1):
    """BP algorithm in log-domain on specific nodes and bits."""

    m, n = H.shape

    var = 10 ** (-snr / 10)

    # step 0: initialization

    Lc = 2 * y / var

    Lq = np.zeros(shape=(m, n))

    Lr = np.zeros(shape=(m, n))

    count = 0

    while(True):

        count += 1
        # step 1 : Horizontal
        for i in range(m):
            ni = bits[i]
            for j in ni:
                nij = ni[:]

                if j in nij:
                    nij.remove(j)

                if count == 1:
                    X = np.prod(np.tanh(0.5 * Lc[nij]))
                else:
                    X = np.prod(np.tanh(0.5*Lq[i, nij]))
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
                if i in mji:
                    mji.remove(i)
                Lq[i, j] = Lc[j]+sum(Lr[mji, j])

        # LLR a posteriori:
        L_posteriori = np.zeros(n)
        for j in range(n):
            mj = nodes[j]

            L_posteriori[j] = Lc[j] + sum(Lr[mj, j])

        x = np.array(L_posteriori <= 0).astype(int)

        product = utils.incode(H, x)

        if product or count >= maxiter:
            break

    if count == maxiter:
        warnings.warn("""Decoding stopped before convergence. You may want
                       to increase maxiter""")
    return x


def get_message(tG, x):
    """Computes the original `n_bits` message from a `n_code` codeword `x`.

    Parameters
    ----------

    tG: array (n_code, n_bits) coding matrix tG.
    x: array (n_code,) decoded codeword of length `n_code`.
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

# @njit(int64[:](int64[:, :], float64[:], float64, int64), cache=True)
