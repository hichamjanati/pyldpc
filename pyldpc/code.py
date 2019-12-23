import numpy as np
from scipy.sparse import csr_matrix
from . import utils


def parity_check_matrix(n, d_v, d_c, seed=None):
    """
    Builds a regular Parity-Check Matrix H following Callager's algorithm.

    Parameters
    ----------
    n: int, Number of coding bits.
    d_v: int, Number of parity-check equations.
    d_c: int, Number of variables in a parity-check equation. d_c Must be
        greater or equal to d_v and must divide n.
    seed: int, seed of the random generator.

    Returns
    -------
    H: array (m, n). Where m = d_v * n / d_c
        LDPC regular matrix H.

    """
    rng = np.random.RandomState(seed)
    if n % d_c:
        raise ValueError("""d_c must divide n for a regular LDPC matrix H.""")

    if d_c <= d_v:
        raise ValueError("""d_c must be greater than d_v.""")

    m = (n * d_v) // d_c

    block = np.zeros((m // d_v, n), dtype=int)
    H = np.empty((m, n))
    block_size = m // d_v

    # Filling the first block with consecutive ones in each row of the block

    for i in range(block_size):
        for j in range(i * d_c, (i+1) * d_c):
            block[i, j] = 1
    H[:block_size] = block

    # reate remaining blocks by permutations of the first block's columns:
    for i in range(1, d_v):
        H[i * block_size: (i + 1) * block_size] = rng.permutation(block.T).T
    H = H.astype(int)
    return H


def coding_matrix(H, sparse=True):

    """
    Returns the generating coding matrix G given the LDPC matrix H.

    Parameters
    ----------

    H: array (m, n). Parity check matrix of an LDPC code with n coding bits.
    sparse: (boolean, default True): if `True`, scipy.sparse format is used
        to speed up computation.

    Returns
    -------

    G.T: array (k, n). Transposed coding matrix.

    """

    if type(H) == csr_matrix:
        H = H.toarray()
    m, n = H.shape

    # DOUBLE GAUSS-JORDAN:

    Href_colonnes, tQ = utils.gaussjordan(H.T, 1)

    Href_diag = utils.gaussjordan(np.transpose(Href_colonnes))

    Q = tQ.T

    k = n - Href_diag.sum()

    Y = np.zeros(shape=(n, k)).astype(int)
    Y[n - k:, :] = np.identity(k)

    if sparse:
        Q = csr_matrix(Q)
        Y = csr_matrix(Y)

    tG = utils.binaryproduct(Q, Y)

    return H, tG


def coding_matrix_systematic(H, sparse=True):

    """Computes a coding matrix G in systematic format with an identity block.

    Parameters
    ----------

    H: array (m, n). Parity-check matrix.
    sparse: (boolean, default True): if `True`, scipy.sparse is used
    to speed up computation if n > 100.

    Returns
    -------

    H_new: (m, n) array. Modified parity-check matrix given by a permutation of
        the columns of the provided H.
    G_systematic.T: Transposed Systematic Coding matrix associated to H_new.

    """

    m, n = H.shape

    if n > 100 and sparse:
        sparse = True
    else:
        sparse = False

    P1 = np.identity(n, dtype=int)

    Hrowreduced = utils.gaussjordan(H)

    k = n - sum([a.any() for a in Hrowreduced])

    # After this loop, Hrowreduced will have the form H_ss : | I_(n-k)  A |

    while(True):
        zeros = [i for i in range(min(m, n)) if not Hrowreduced[i, i]]
        indice_colonne_a = min(zeros)
        list_ones = [j for j in range(indice_colonne_a+1, n)
                     if Hrowreduced[indice_colonne_a, j]]
        if not len(list_ones):
            break

        indice_colonne_b = min(list_ones)

        aux = Hrowreduced[:, indice_colonne_a].copy()
        Hrowreduced[:, indice_colonne_a] = Hrowreduced[:, indice_colonne_b]
        Hrowreduced[:, indice_colonne_b] = aux

        aux = P1[:, indice_colonne_a].copy()
        P1[:, indice_colonne_a] = P1[:, indice_colonne_b]
        P1[:, indice_colonne_b] = aux

    # NOW, Hrowreduced has the form: | I_(n-k)  A | ,
    # the permutation above makes it look like :
    # |A  I_(n-k)|

    P1 = P1.T
    identity = list(range(n))
    sigma = identity[n-k:] + identity[:n-k]

    P2 = np.zeros(shape=(n, n), dtype=int)
    P2[identity, sigma] = np.ones(n)

    if sparse:
        P1 = csr_matrix(P1)
        P2 = csr_matrix(P2)
        H = csr_matrix(H)

    P = utils.binaryproduct(P2, P1)

    if sparse:
        P = csr_matrix(P)

    H_new = utils.binaryproduct(H, np.transpose(P))

    G_systematic = np.zeros((k, n), dtype=int)
    G_systematic[:, :k] = np.identity(k)
    G_systematic[:, k:] = (Hrowreduced[:n-k, n-k:]).T

    return H_new, G_systematic.T


def make_ldpc(n, d_v, d_c, systematic=False, sparse=True, seed=None):
    """Creates an LDPC coding and decoding matrices H and G.
    Parameters:
    -----------
    Parameters
    ----------
    n: int, Number of coding bits.
    d_v: int, Number of parity-check equations.
    d_c: int, Number of variables in a parity-check equation. d_c Must be
        greater or equal to d_v and must divide n.
    systematic: boolean, default False. if True, constructs a systematic
    coding matrix G.

    Returns:
    --------
    H: (n, m) array with m.d_c = n.d_v with m < n. parity check matrix
    G: (n, k) array coding matrix."""

    H = parity_check_matrix(n, d_v, d_c, seed=seed)
    if systematic:
        H, G = coding_matrix_systematic(H, sparse=sparse)
    else:
        H, G = coding_matrix(H, sparse=sparse)
    return H, G
