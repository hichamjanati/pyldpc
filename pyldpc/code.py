import numpy as np
from scipy.sparse import csr_matrix
from . import utils


def parity_check_matrix(n, d_v, d_c, seed=None):
    """
    Builds a regular Parity-Check Matrix H (n, d_v, d_c) following
    Callager's algorithm.

    Parameters:

     n: Number of columns (Same as number of coding bits)
     d_v: number of ones per column (number of parity-check equations including
     a certain variable)
     d_c: number of ones per row (number of variables participating in a
     certain parity-check equation);

    Errors:

     The number of ones in the matrix is the same no matter how we calculate
     it (rows or columns), therefore, if m is
     the number of rows in the matrix:

     m*d_c = n*d_v with m < n (because H is a decoding matrix) => Parameters
     must verify:


     0 - all integer parameters
     1 - d_v < d_v
     2 - d_c divides n

    ---------------------------------------------------------------------------------------

     Returns: 2D-array (shape = (m, n))

    """
    rnd = np.random.RandomState(seed)
    if n % d_c:
        raise ValueError("""d_c must divide n. help(coding_matrix)
                            for more info.""")

    if d_c <= d_v:
        raise ValueError("""d_c must be greater than d_v.
                            help(coding_matrix) for
                            more info.""")

    m = (n * d_v) // d_c

    Set = np.zeros((m//d_v, n), dtype=int)
    a = m // d_v

    # Filling the first set with consecutive ones in each row of the set

    for i in range(a):
        for j in range(i * d_c, (i+1 * d_c)):
            Set[i, j] = 1

    # Create list of Sets and append the first reference set
    Sets = []
    Sets.append(Set.tolist())

    # reate remaining sets by permutations of the first set's columns:
    i = 1
    for i in range(1, d_v):
        newSet = rnd.permutation(np.transpose(Set)).T.tolist()
        Sets.append(newSet)

    # Returns concatenated list of sest:
    H = np.concatenate(Sets)
    return H


def coding_matrix(X, sparse=True):

    """
    CAUTION: RETURNS tG TRANSPOSED CODING X.
    Function Applies gaussjordan Algorithm on Columns and rows of X in
    order to permute Basis Change matrix using Matrix Equivalence.
    Let A be the treated Matrix. refAref the double row reduced echelon Matrix.

    refAref has the form:

    (e.g) : |1 0 0 0 0 0 ... 0 0 0 0|
            |0 1 0 0 0 0 ... 0 0 0 0|
            |0 0 0 0 0 0 ... 0 0 0 0|
            |0 0 0 1 0 0 ... 0 0 0 0|
            |0 0 0 0 0 0 ... 0 0 0 0|
            |0 0 0 0 0 0 ... 0 0 0 0|


    First, let P1 Q1 invertible matrices: P1.A.Q1 = refAref

    We would like to calculate:
    P,Q are the square invertible matrices of the appropriate size so that:

    P.A.Q = J.  Where J is the matrix of the form (having X's shape):

    | I_p O | where p is X's rank and I_p Identity matrix of size p.
    | 0   0 |

    Therfore, we perform permuations of rows and columns in refAref
    (same changes are applied to Q1 in order to get final Q matrix)

    NOTE: P IS NOT RETURNED BECAUSE WE DO NOT NEED IT TO SOLVE H.G' = 0
    P IS INVERTIBLE, WE GET SIMPLY RID OF IT.

    Then

    solves: inv(P).J.inv(Q).G' = 0 (1) where inv(P) = P^(-1) and
    P.H.Q = J. Help(PJQ) for more info.

    Let Y = inv(Q).G', equation becomes J.Y = 0 (2) whilst:

    J = | I_p O | where p is H's rank and I_p Identity matrix of size p.
        | 0   0 |

    Knowing that G must have full rank, a solution of (2)
    is Y = |  0  | Where k = n-p.
           | I-k |

    Because of rank-nullity theorem.

    -----------------
    parameters:

    H: Parity check matrix.
    sparse: (optional, default True): use scipy.sparse format to speed up
    computation.
    ---------------
    returns:

    tG: Transposed Coding Matrix.

    """

    H = np.copy(X)
    m, n = H.shape

    # DOUBLE GAUSS-JORDAN:

    Href_colonnes, tQ = utils.gaussjordan(np.transpose(H), 1)

    Href_diag = utils.gaussjordan(np.transpose(Href_colonnes))

    Q = np.transpose(tQ)

    k = n - sum(Href_diag.reshape(m*n))

    Y = np.zeros(shape=(n, k)).astype(int)
    Y[n-k:, :] = np.identity(k)

    if sparse:
        Q = csr_matrix(Q)
        Y = csr_matrix(Y)

    tG = utils.binaryproduct(Q, Y)

    return H, tG


def coding_matrix_systematic(X, sparse=True):

    """
    Description:

    Solves H.G' = 0 and finds the coding matrix G in the systematic form :
    [I_k  A] by applying permutations on X.

    CAUTION: RETURNS TUPLE (Hp,tGS) WHERE Hp IS A MODIFIED VERSION OF THE
    GIVEN PARITY CHECK X, tGS THE TRANSPOSED
    SYSTEMATIC CODING X ASSOCIATED TO Hp. YOU MUST USE THE RETURNED TUPLE
    IN CODING AND DECODING, RATHER THAN THE UNCHANGED
    PARITY-CHECK X H.

    -------------------------------------------------
    Parameters:

    X: 2D-Array. Parity-check matrix.
    sparse: (optional, default True): use scipy.sparse matrices
    to speed up computation if n>100.

    ------------------------------------------------

    >>> Returns Tuple of 2D-arrays (Hp,GS):
        Hp: Modified H: permutation of columns (The code doesn't change)
        tGS: Transposed Systematic Coding matrix associated to Hp.

    """

    H = X.copy()
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

    Hp = utils.binaryproduct(H, np.transpose(P))

    GS = np.zeros((k, n), dtype=int)
    GS[:, :k] = np.identity(k)
    GS[:, k:] = (Hrowreduced[:n-k, n-k:]).T

    return Hp, GS.T


def make_ldpc(n, d_v, d_c, seed=None, systematic=False, sparse=True):
    """Creates an LDPC coding and decoding matrices H and G.
    Parameters:
    -----------
    n: Number of columns (same as number of coding bits)
    d_v: number of ones per column (number of parity-check equations including
    a certain variable)
    d_c: number of ones per row (number of variables participating in a
    certain parity-check equation);
    systematic: optional, default False. if True, constructs a systematic
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
