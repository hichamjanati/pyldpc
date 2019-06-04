import numpy as np
import warnings
from . import utils


def decode(H, y, snr, maxiter=100, log=True):
    """Decoder function."""
    f = decode_bp
    if log:
        f = decode_logbp
    return f(H, y, snr, maxiter=maxiter)


def decode_bp(H, y, snr, maxiter=100):

    """ Decoding function using Belief Propagation algorithm.
        IMPORTANT: H can be scipy.sparse.csr_matrix object to speed up
        calculations if n > 1000 highly recommanded.
    -----------------------------------
    Parameters:

    H: 2D-array (m, n) (OR scipy.sparse.csr_matrix object) Parity check matrix.

    y: n-vector recieved after transmission in the channel. (In general,
    returned by Coding Function)

    Signal-Noise Ratio: SNR = 10log(1/variance) in decibels of the AWGN used
    in coding.

    max_iter: (default = 1) max iterations of the main loop. Increase if
    decoding is not error-free.

     """

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


def decode_logbp(H, y, snr, maxiter=1):

    """ Decoding function using Belief Propagation algorithm
    (logarithmic version)
    IMPORTANT: if H is large (n>1000), H should be scipy.sparse.csr_matrix
    object to speed up calculations
    (highly recommanded. )
    -----------------------------------
    Parameters:

    H: 2D-array (m, n) (OR scipy.sparse.csr_matrix object) Parity check matrix
    y: n-vector recieved after transmission in the channel. (In particular,
    returned by `encode` Function)
    Signal-Noise Ratio: SNR = 10log(1/variance) in decibels of the AWGN
    used in coding.

    maxiter: (default = 1) max iterations of the main loop.
    Increase if decoding is not error-free.

     """

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


def decode_bp_ext(H, bits, nodes, y, snr, maxiter=1):

    """ Decoding function using Belief Propagation algorithm.

        IMPORTANT: H can be scipy.sparse.csr_matrix object to speed
        up calculations if n > 1000 highly recommanded.
    -----------------------------------
    Parameters:

    H: 2D-array (OR scipy.sparse.csr_matrix object) Parity check matrix,
    shape = (m, n)
    bits, nodes: returned by `bitsandnodes` function.
    y: n-vector recieved after transmission in the channel. (In general,
    returned by `coding` Function)
    Signal-Noise Ratio: SNR = 10log(1/variance) in decibels of the AWGN
    used in coding.
    maxiter: (default = 1) max iterations of the main loop.
    Increase if decoding is not error-free.

     """

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


def decode_logbp_ext(H, bits, nodes, y, snr, maxiter=1):

    """ Decoding function using Belief Propagation algorithm.
    in logdomain.

        IMPORTANT: H can be scipy.sparse.csr_matrix object to speed
        up calculations if n > 1000 highly recommanded.
    -----------------------------------
    Parameters:

    H: 2D-array (OR scipy.sparse.csr_matrix object) Parity check matrix,
    shape = (m, n)
    bits, nodes: returned by `bitsandnodes` function.
    y: n-vector recieved after transmission in the channel. (In general,
    returned by `coding` Function)
    Signal-Noise Ratio: SNR = 10log(1/variance) in decibels of the AWGN
    used in coding.
    maxiter: (default = 1) max iterations of the main loop.
    Increase if decoding is not error-free.

     """

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

    """
    Let G be a coding matrix. tG its transposed matrix.
    x a n-vector received after decoding.
    DecodedMessage Solves the equation on k-bits message v:
    x = v.G => G'v'= x' by applying GaussElimination on G'.

    -------------------------------------

    Parameters:

    tG: Transposed Coding Matrix. Must have more rows than columns to solve
    the linear system. Must be full rank.
    x: n-array. Must be in the Code (in Ker(H)).

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
