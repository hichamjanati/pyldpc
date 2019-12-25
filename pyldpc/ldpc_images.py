import numpy as np
from .utils_img import (bin2gray, bin2rgb)
from .encoder import encode
from .decoder import get_message, decode
from .utils import bitsandnodes, check_random_state
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
    coded_img: array (n, n_blocks) image in the codeword space
    noisy_img: array (height, width, k) visualization of the noisy image

    """
    seed = check_random_state(seed)
    n, k = tG.shape

    height, width, depth = img_bin.shape
    if depth not in [8, 24]:
        raise ValueError("The expected dimension of a binary image is "
                         "(width, height, 8) for grayscale images or "
                         "(width, height, 24) for RGB images; got %s"
                         % list(img_bin.shape))
    img_bin = img_bin.flatten()
    n_bits_total = img_bin.size
    n_blocks = n_bits_total // k
    residual = n_bits_total % k
    if residual:
        n_blocks += 1
    resized_img = np.zeros(k * n_blocks)
    resized_img[:n_bits_total] = img_bin

    codeword = encode(tG, resized_img.reshape(k, n_blocks), snr, seed)
    noisy_img = (codeword.flatten()[:n_bits_total] < 0).astype(int)
    noisy_img = noisy_img.reshape(width, height, depth)

    if depth == 8:
        noisy_img = bin2gray(noisy_img)
    else:
        noisy_img = bin2rgb(noisy_img)

    return codeword, noisy_img


def decode_img(tG, H, codeword, snr, img_shape, maxiter=10000):
    """Decode a received noisy image in the codeword.

    Parameters
    ----------
    tG: array (n, k) coding matrix G
    H: array (m, n) decoding matrix H
    img_coded: array (n, n_blocks) image recieved in the codeword
    snr: float. signal to noise ratio assumed of the channel.
    img_shape: tuple of int. Shape of the original binary image.
    maxiter: int. Max number of BP iterations to perform.
    n_jobs: int. Number of parallel jobs.

    Returns
    -------
    img_decode: array(width, height, depth). Decoded image.

    """
    n, k = tG.shape
    _, n_blocks = codeword.shape

    depth = img_shape[-1]
    if depth not in [8, 24]:
        raise ValueError("The expected dimension of a binary image is "
                         "(width, height, 8) for grayscale images or "
                         "(width, height, 24) for RGB images; got %s"
                         % list(img_shape))
    if len(codeword) != n:
        raise ValueError("The left dimension of `codeword` must be equal to "
                         "n, the number of columns of H.")

    systematic = True

    if not (tG[:k, :] == np.identity(k)).all():
        warnings.warn("""In LDPC applications, using systematic coding matrix
                         G is highly recommanded to speed up decoding.""")
        systematic = False

    bits, nodes = bitsandnodes(H)

    codeword_solution = decode(H, codeword, snr, maxiter)
    if systematic:
        decoded = codeword_solution[:k, :]
    else:
        decoded = np.array([get_message(tG, codeword_solution[:, i])
                           for i in range(n_blocks)]).T
    decoded = decoded.flatten()[:np.prod(img_shape)]
    decoded = decoded.reshape(*img_shape)

    if depth == 8:
        decoded_img = bin2gray(decoded)
    else:
        decoded_img = bin2rgb(decoded)

    return decoded_img


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
