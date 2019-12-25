"""Coding module."""
import numpy as np
from scipy.sparse import csr_matrix
from . import utils


def parity_check_matrix(n_code, d_v, d_c, seed=None):
    """
    Build a regular Parity-Check Matrix H following Callager's algorithm.

    Parameters
    ----------
    n_code: int, Length of the codewords.
    d_v: int, Number of parity-check equations including a certain bit.
        Must be greater or equal to 2.
    d_c: int, Number of bits in the same parity-check equation. d_c Must be
        greater or equal to d_v and must divide n.
    seed: int, seed of the random generator.

    Returns
    -------
    H: array (n_equations, n_code). LDPC regular matrix H.
        Where n_equations = d_v * n / d_c, the total number of parity-check
        equations.

    """
    rng = utils.check_random_state(seed)

    if d_v <= 1:
        raise ValueError("""d_v must be at least 2.""")

    if d_c <= d_v:
        raise ValueError("""d_c must be greater than d_v.""")

    if n_code % d_c:
        raise ValueError("""d_c must divide n for a regular LDPC matrix H.""")

    n_equations = (n_code * d_v) // d_c

    block = np.zeros((n_equations // d_v, n_code), dtype=int)
    H = np.empty((n_equations, n_code))
    block_size = n_equations // d_v

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
    """Return the generating coding matrix G given the LDPC matrix H.

    Parameters
    ----------
    H: array (n_equations, n_code). Parity check matrix of an LDPC code with
        code length `n_code` and `n_equations` number of equations.
    sparse: (boolean, default True): if `True`, scipy.sparse format is used
        to speed up computation.

    Returns
    -------
    G.T: array (n_bits, n_code). Transposed coding matrix.

    """
    if type(H) == csr_matrix:
        H = H.toarray()
    n_equations, n_code = H.shape

    # DOUBLE GAUSS-JORDAN:

    Href_colonnes, tQ = utils.gaussjordan(H.T, 1)

    Href_diag = utils.gaussjordan(np.transpose(Href_colonnes))

    Q = tQ.T

    n_bits = n_code - Href_diag.sum()

    Y = np.zeros(shape=(n_code, n_bits)).astype(int)
    Y[n_code - n_bits:, :] = np.identity(n_bits)

    if sparse:
        Q = csr_matrix(Q)
        Y = csr_matrix(Y)

    tG = utils.binaryproduct(Q, Y)

    return H, tG


def coding_matrix_systematic(H, sparse=True):
    """Compute a coding matrix G in systematic format with an identity block.

    Parameters
    ----------
    H: array (n_equations, n_code). Parity-check matrix.
    sparse: (boolean, default True): if `True`, scipy.sparse is used
    to speed up computation if n_code > 1000.

    Returns
    -------
    H_new: (n_equations, n_code) array. Modified parity-check matrix given by a
        permutation of the columns of the provided H.
    G_systematic.T: Transposed Systematic Coding matrix associated to H_new.

    """
    n_equations, n_code = H.shape

    if n_code > 1000 and sparse:
        sparse = True
    else:
        sparse = False

    P1 = np.identity(n_code, dtype=int)

    Hrowreduced = utils.gaussjordan(H)

    n_bits = n_code - sum([a.any() for a in Hrowreduced])

    # After this loop, Hrowreduced will have the form H_ss : | I_(n-k)  A |

    while(True):
        zeros = [i for i in range(min(n_equations, n_code))
                 if not Hrowreduced[i, i]]
        indice_colonne_a = min(zeros)
        list_ones = [j for j in range(indice_colonne_a + 1, n_code)
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
    identity = list(range(n_code))
    sigma = identity[n_code - n_bits:] + identity[:n_code - n_bits]

    P2 = np.zeros(shape=(n_code, n_code), dtype=int)
    P2[identity, sigma] = np.ones(n_code)

    if sparse:
        P1 = csr_matrix(P1)
        P2 = csr_matrix(P2)
        H = csr_matrix(H)

    P = utils.binaryproduct(P2, P1)

    if sparse:
        P = csr_matrix(P)

    H_new = utils.binaryproduct(H, np.transpose(P))

    G_systematic = np.zeros((n_bits, n_code), dtype=int)
    G_systematic[:, :n_bits] = np.identity(n_bits)
    G_systematic[:, n_bits:] = \
        (Hrowreduced[:n_code - n_bits, n_code - n_bits:]).T

    return H_new, G_systematic.T


def make_ldpc(n_code, d_v, d_c, systematic=False, sparse=True, seed=None):
    """Create an LDPC coding and decoding matrices H and G.

    Parameters
    ----------
    n_code: int, Length of the codewords.
    d_v: int, Number of parity-check equations including a certain bit.
    d_c: int, Number of bits in the same parity-check equation. d_c Must be
        greater or equal to d_v and must divide n.
    seed: int, seed of the random generator.
    systematic: boolean, default False. if True, constructs a systematic
    coding matrix G.

    Returns:
    --------
    H: array (n_equations, n_code). Parity check matrix of an LDPC code with
        code length `n_code` and `n_equations` number of equations.
    G: (n_code, n_bits) array coding matrix.

    """
    seed = utils.check_random_state(seed)

    H = parity_check_matrix(n_code, d_v, d_c, seed=seed)
    if systematic:
        H, G = coding_matrix_systematic(H, sparse=sparse)
    else:
        H, G = coding_matrix(H, sparse=sparse)
    return H, G
