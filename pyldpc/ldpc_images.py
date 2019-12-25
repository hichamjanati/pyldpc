import numpy as np
from .utils_img import (bin2gray, bin2rgb)
from .encoder import encode
from .decoder import _decode_logbp_ext, get_message
from .utils import bitsandnodes
import scipy
import warnings


def encode_img(tG, img_bin, snr, seed=None):
    """Encode a binary image and adds Gaussian white noise.

    Parameters
    ----------
    tG: array (n, k). Coding matrix. `k` is the number of bits to be coded.
        `n` is the length of the codewords.
    img_bin: array (height, width, depth). Binary image.
    snr : float. Signal to noise ratio of the channel.
    seed: int. random state initialization.

    Returns
    -------
    coded_img: array (height, width, n) image in the codeword space
    noisy_img: array (height, width, k) visualization of the noisy image

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


def decode_img(tG, H, img_coded, snr, maxiter=1000):
    """Decode a received noisy image in the codeword.

    Parameters
    ----------
    tG: array (n, k) coding matrix G
    H: array (m, n) decoding matrix H
    img_coded: array (width, height, depth) image recieved in the codeword
    snr: float. signal to noise ratio assumed of the channel.
    maxiter: int. Max number of BP iterations to perform.

    Returns
    -------
    img_decode: array(width, height, depth). Decoded image.

    """
    n, k = tG.shape
    height, width, depth = img_coded.shape

    img_decoded_bin = np.zeros(shape=(height, width, k), dtype=int)

    decodefunction = _decode_logbp_ext

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
    """Encode an image by grouping pixels row by row.

    Parameters
    ----------
    tG: array (n, k) coding matrix G
    img_bin: array (width, height, depth_) image to be coded.
    snr: float. signal to noise ratio assumed of the channel.
    seed: int. random state initialization.

    Returns
    -------
    coded_img: array (_) image in the codeword space
    noisy_img: array (_) visualization of the noisy image

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


def decode_img_rowbyrow(tG, H, img_coded, snr, maxiter=1000):
    """Decode a noisy image in the codeword by grouping pixels in rows.

    Parameters
    ----------
    tG: array (n, k) coding matrix G
    H: array (m, n) decoding matrix H
    img_coded: array (width, height, depth) image recieved in the codeword
    snr: float. signal to noise ratio assumed of the channel.
    maxiter: int. Max number of BP iterations to perform.

    Returns
    -------
    img_decode: array(width, height, depth). Decoded image.

    """
    n, k = tG.shape
    width, depth = img_coded.astype(int)[-1, 0:2]
    img_coded = img_coded[:-1, :]
    height, N = img_coded.shape

    if N != n:
        raise ValueError("""Coded Image must have the same number of
                            columns as H""")
    if depth != 8 and depth != 24:
        raise ValueError("""Type of image not recognized: third dimension of
                         the binary image must be 8 for grayscale,
                         or 24 for rgb images""")

    if not (tG[:k, :] == np.identity(k)).all():
        raise ValueError("""G must be Systematic. Solving tG.tv = tx for images
                            has a O(n^3) complexity""")

    if not type(H) == scipy.sparse.csr_matrix:
        warnings.warn("""The matrix H provided is not a csr object. Using
                         scipy.sparse.csr_matrix format is highly
                         recommanded when doing row by row coding and
                         decoding to speed up calculations.""")

    img_decoded_bin = np.zeros(shape=(height, k), dtype=int)

    decodefunction = _decode_logbp_ext
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
    """Compute Bit-Error-Rate (BER) by comparing 2 binary images."""
    if not original_img_bin.shape == decoded_img_bin.shape:
        raise ValueError('Original and decoded images\' shapes don\'t match !')

    height, width, k = original_img_bin.shape

    errors_bits = abs(original_img_bin - decoded_img_bin).sum()
    errors_bits = errors_bits.flatten()
    total_bits = np.prod(original_img_bin.shape)

    ber = errors_bits / total_bits

    return(ber)
