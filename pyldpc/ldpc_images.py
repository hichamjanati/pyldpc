import numpy as np
from .utils_img import (bin2gray, bin2rgb)
from .encoder import encode
from .decoder import (decode_bp_ext, decode_logbp_ext,
                      get_message)
from .utils import bitsandnodes
import scipy
import warnings


def encode_img(tG, img_bin, snr, seed=None):

    """
    CAUTION: SINCE V.0.7 Image coding and `decode` functions
    TAKES TRANSPOSED CODING MATRIX tG.
    IF G IS LARGE, USE SCIPY.SPARSE.CSR_MATRIX FORMAT TO SPEED UP CALCULATIONS.
    Codes a binary image (Therefore must be a 3D-array).
    Each pixel (k bits-array, k=8 if grayscale, k=24 if colorful)
    is considered a k-bits message. If the original binary image is
    shaped (X,Y, k). The coded image will be shaped (X,Y, n)
    Where n is the length of a codeword.
    Then a gaussian noise N(0, snr) is added to the codeword.

    Remember SNR: Signal-Noise Ratio: SNR = 10log(1/variance) in decibels
    of the AWGN used in coding.

    Of course, showing an image with n-bits array is impossible,
    that's why if the optional argument show is set to 1,
    and if Coding Matrix G is systematic, showing the noisy
    image can be possible by gathering the k first bits of each
    n-bits codeword to the left, and the redundant bits to the right.
    Then the noisy image is changed from bin to uint8.
    Remember that in this case, ImageCoding returns  a tuple: the (X,Y, n)
    coded image, and the noisy image (X,Y*(n//k)).

    Parameters:

    G: Coding Matrix G - (must be systematic to see what the noisy image
    looks like.) See CodingMatrix_systematic.
    img_bin: 3D-array of a binary image.
    SNR: Signal-Noise Ratio, SNR = 10log(1/variance) in decibels of
    the AWGN used in coding.

    Returns:
    (default): Tuple: noisy_img, coded_img

    """

    n, k = tG.shape

    height, width, depth = img_bin.shape

    if k != 8 and k != 24:
        raise ValueError("""Coding matrix must have 8 xor 24 rows
                         ( grayscale images xor rgb images)""")

    coded_img = np.zeros(shape=(height, width, n))

    noisy_img = np.zeros(shape=(height, width, k), dtype=int)

    for i in range(height):
        for j in range(width):
            coded_byte_ij = encode(tG, img_bin[i, j, :], snr, seed)
            coded_img[i, j, :] = coded_byte_ij
            systematic_part_ij = (coded_byte_ij[:k] < 0).astype(int)

            noisy_img[i, j, :] = systematic_part_ij

    if k == 8:
        noisy_img = bin2gray(noisy_img)
    else:
        noisy_img = bin2rgb(noisy_img)

    return coded_img, noisy_img


def decode_img(tG, H, img_coded, snr, maxiter=100, log=True):
    """
    CAUTION: SINCE V.0.7 Image coding and decode functions TAKES TRANSPOSED
    CODING MATRIX tG.

    IF G IS LARGE, USE SCIPY.SPARSE.CSR_MATRIX FORMAT (IN H AND G) TO SPEED UP
    CALCULATIONS.

    Image decode Function. Taked the 3-D binary coded image where each
    element is a codeword n-bits array and decodes
    every one of them. Needs H to decode. A k bits decoded vector is the
    first k bits of each codeword, the decoded image can
    be transformed from binary to uin8 format and shown.

    Parameters:

    tG: Transposed coding matrix tG.
    H: Parity-Check Matrix (decode matrix).
    img_coded: binary coded image returned by the function ImageCoding.
    Must be shaped (heigth, width, n) where n is a
                the length of a codeword (also the number of H's columns)

    snr: Signal-Noise Ratio: SNR = 10log(1/variance) in decibels of the
    AWGN used in coding.

    log: (optional, default = True), if True, Full-log version of BP
    algorithm is used.
    maxiter: (optional, default =1), number of iterations of decode.
    increase if snr is < 5db.
    """
    n, k = tG.shape
    height, width, depth = img_coded.shape

    img_decoded_bin = np.zeros(shape=(height, width, k), dtype=int)

    if log:
        decodefunction = decode_logbp_ext
    else:
        decodefunction = decode_bp_ext

    systematic = True

    if not (tG[:k, :] == np.identity(k)).all():
        warnings.warn("""In LDPC applications, using systematic coding matrix
                         G is highly recommanded to speed up decode.""")
        systematic = False

    bits, nodes = bitsandnodes(H)
    for i in range(height):
        for j in range(width):
            decoded_vector = decodefunction(H, bits, nodes, img_coded[i, j, :],
                                            snr, maxiter)
            if systematic:
                decoded_byte = decoded_vector[:k]
            else:
                decoded_byte = get_message(tG, decoded_vector)

            img_decoded_bin[i, j, :] = decoded_byte

    if k == 8:
        img_decoded = bin2gray(img_decoded_bin)
    else:
        img_decoded = bin2rgb(img_decoded_bin)

    return img_decoded


def encode_img_rowbyrow(tG, img_bin, snr, seed=None):

    """

    CAUTION: SINCE V.0.7 Image coding and decode functions TAKE TRANSPOSED
    CODING MATRIX tG. USE
    SCIPY.SPARSE.CSR_MATRIX FORMAT (IN H AND G) TO SPEED UP CALCULATIONS.
    K MUST BE EQUAL TO THE NUMBER OF BITS IN ONE ROW
    OF THE BINARY IMAGE. USE A SYSTEMATIC CODING MATRIX WITH
    CodingMatrix_systematic. THEN USE SCIPY.SPARSE.CSR_MATRIX()

    --------

    Codes a binary image (Therefore must be a 3D-array). Each row of img_bin
    is considered a k-bits message. If the image has
    a shape (X,Y,Z) then the binary image will have the shape (X, k).
    The coded image will be shaped (X+1, n):

    - The last line of the coded image stores Y and Z so as to be able to
    construct the decoded image again via a reshape.
    - n is the length of a codeword. Then a gaussian noise N(0, snr)
    is added to the codeword.

    Remember SNR: Signal-Noise Ratio: SNR = 10log(1/variance) in decibels
    of the AWGN used in coding.

    ImageCoding returns  a tuple: the (X+1, n) coded image, and the noisy
    image (X, Y, Z).

    Parameters:

    tG: Transposed Coding Matrix G - must be systematic.
    See codingmatrix_systematic.
    img_bin: 3D-array of a binary image.
    SNR: Signal-Noise Ratio, SNR = 10log(1/variance) in decibels of the AWGN
    used in coding.

    Returns:
    (default): Tuple:  coded_img, noisy_img

    """
    n, k = tG.shape
    height, width, depth = img_bin.shape

    if not type(tG) == scipy.sparse.csr_matrix:
        warnings.warn("""Using scipy.sparse.csr_matrix format is highly
                    recommanded when computing row by row coding and decode
                    to speed up calculations.""")

    if not (tG[:k, :] == np.identity(k)).all():
        raise ValueError("""G must be Systematic. Solving tG.tv = tx for images
                            has a O(n^3) complexity.""")

    if width * depth != k:
        raise ValueError("""If the image's shape is (X,Y,Z) k must be equal
                         to 8*Y (if gray ) or 24*Y (if rgb)""")

    img_bin_reshaped = img_bin.reshape(height, width*depth)

    coded_img = np.zeros(shape=(height+1, n))
    coded_img[height, 0:2] = width, depth

    for i in range(height):
        coded_img[i, :] = encode(tG, img_bin_reshaped[i, :], snr, seed)

    noisy_img = (coded_img[:height, :k] < 0).astype(int)
    noisy_img = noisy_img.reshape(height, width, depth)

    if depth == 8:
        return coded_img, bin2gray(noisy_img)
    if depth == 24:
        return coded_img, bin2rgb(noisy_img)


def decode_img_rowbyrow(tG, H, img_coded, snr, maxiter=1, log=1):

    """
    CAUTION: SINCE V.0.7 Imagedecode TAKES TRANSPOSED CODING MATRIX
    tG INSTEAD OF G. USE SCIPY.SPARSE.CSR_MATRIX
    FORMAT (IN H AND G) TO SPEED UP CALCULATIONS.

    --------
    Image decode Function. Taked the 3-D binary coded image where each
    element is a codeword n-bits array and decodes
    every one of them. Needs H to decode and tG to solve tG.tv = tx where
    x is the codeword element decoded by the function
    itself. When v is found for each codeword, the decoded image can be
    transformed from binary to uin8 format and shown.

    Parameters:

    tG: Transposed Coding Matrix ( SCIPY.SPARSE.CSR_MATRIX FORMAT RECOMMANDED )
    H: Parity-Check Matrix (decode matrix).( SCIPY.SPARSE.CSR_MATRIX
    FORMAT RECOMMANDED)

    img_coded: binary coded image returned by the function ImageCoding.
    Must be shaped (heigth, width, n) where n is the
    length of a codeword (also the number of H's columns)

    snr: Signal-Noise Ratio: SNR = 10log(1/variance) in decibels of the AWGN
    used in coding.

    log: (optional, default = True), if True, Full-log version of BP algorithm
    is used.
    maxiter: (optional, default =1), number of iterations of decode.
    increase if snr is < 5db.
    """

    n, k = tG.shape
    width, depth = img_coded.astype(int)[-1, 0:2]
    img_coded = img_coded[:-1, :]
    height, N = img_coded.shape

    if N != n:
        raise ValueError("""Coded Image must have the same number of
                            columns as H""")
    if depth != 8 and depth != 24:
        raise ValueError("""type of image not recognized: third dimension of
                         the binary image must be 8 for grayscale,
                         or 24 for rgb images""")

    if not (tG[:k, :] == np.identity(k)).all():
        raise ValueError("""G must be Systematic. Solving tG.tv = tx for images
                            has a O(n^3) complexity""")

    if not type(H) == scipy.sparse.csr_matrix:
        warnings.warn("""Used H is not a csr object. Using
                         scipy.sparse.csr_matrix format is highly
                         recommanded when computing row by row coding and
                         decode to speed up calculations.""")

    img_decoded_bin = np.zeros(shape=(height, k), dtype=int)

    if log:
        decodefunction = decode_logbp_ext
    else:
        decodefunction = decode_bp_ext

    bits, nodes = bitsandnodes(H)

    for i in range(height):
        decoded_vector = decodefunction(H, bits, nodes, img_coded[i, :], snr,
                                        maxiter)
        img_decoded_bin[i, :] = decoded_vector[:k]

    if depth == 8:
        img_decoded = bin2gray(img_decoded_bin.reshape(height, width, depth))
    if depth == 24:
        img_decoded = bin2rgb(img_decoded_bin.reshape(height, width, depth))

    return img_decoded


def ber_img(original_img_bin, decoded_img_bin):
    """

    Computes Bit-Error-Rate (BER) by comparing 2 binary images.
    The ratio of bit errors over total number of bits is returned.

    """
    if not original_img_bin.shape == decoded_img_bin.shape:
        raise ValueError('Original and decoded images\' shapes don\'t match !')

    height, width, k = original_img_bin.shape

    errors_bits = abs(original_img_bin - decoded_img_bin).sum()
    errors_bits = errors_bits.flatten()
    total_bits = np.prod(original_img_bin.shape)

    ber = errors_bits / total_bits

    return(ber)
