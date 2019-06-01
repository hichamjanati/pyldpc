import numpy as np
import warnings

from .utils_audio import bin2audio
from .encoder import encode
from .decoder import (decode_bp_ext, decode_logbp_ext,
                      get_message)
from .utils import bitsandnodes


def encode_audio(tG, audio_bin, snr, seed=None):
    """
    Codes a binary audio array (Therefore must be a 2D-array shaped
    (length,17)). Each element (17 bits)
    is considered a k-bits message. If the original binary array is shaped
    (length,17). The coded image will be shaped
    (length, n) Where n is the length of a codeword.
    Then a gaussian noise N(0, snr) is added to the codeword.

    Remember SNR: Signal-Noise Ratio: SNR = 10log(1/variance) in decibels of
    the AWGN used in coding.

    Of course, "listening" to an audio file with n-bits array is impossible,
    that's why if coding Matrix G is systematic,
    reading the noisy sound can be possible by gathering the 17 first bits
    of each
    n-bits codeword to the left, the redundant bits are dropped.
    Then the noisy sound is changed from bin to int16.
    returns  a tuple: the (length, n) coded audio, and the noisy one (length).

    Parameters:

    tG: Transposed coding Matrix G - must be systematic if you want to see
    what the noisy audio sounds like. See codingMatrix_systematic.
    audio_bin: 2D-array of a binary audio shaped (length,17).
    SNR: Signal-Noise Ratio, SNR = 10log(1/variance) in decibels of the
    AWGN used in coding.

    Returns:
    Tuple: noisy_audio, coded_audio
    """

    n, k = tG.shape
    length = audio_bin.shape[0]

    if k != 17:
        raise ValueError("""coding matrix G must have 17 rows (audio files are
                            written in int16 which is equivalent to uint17)""")

    coded_audio = np.zeros(shape=(length, n))

    noisy_audio = np.zeros(shape=(length, k), dtype=int)

    for j in range(length):
        coded_number_j = encode(tG, audio_bin[j, :], snr, seed)
        coded_audio[j, :] = coded_number_j
        systematic_part_j = (coded_number_j[:k] < 0).astype(int)
        noisy_audio[j, :] = systematic_part_j

    noisy_audio = bin2audio(noisy_audio)

    return coded_audio, noisy_audio


def decode_audio(tG, H, audio_coded, snr, maxiter=1, log=True):

    """
    Sound decode function. Taked the 2-D binary coded audio array where
    each element is a codeword n-bits array and decodes
    every one of them. Needs H to decode and G to solve v.G = x where x is
    the codeword element decoded by the function
    itself. When v is found for each codeword, the decoded audio can be
    transformed from binary to int16 format and read.

    Parameters:

    tG: Transposed coding Matrix
    H: Parity-Check Matrix (decode matrix).
    audio_coded: binary coded audio returned by the function Soundcoding.
    Must be shaped (length, n) where n is a
                the length of a codeword (also the number of H's columns)

    snr: Signal-Noise Ratio: SNR = 10log(1/variance) in decibels of the
    AWGN used in coding.

    log: (optional, default = True), if True, Full-log version of bp
    algorithm is used.
    maxiter: (optional, default =1), number of iterations of decode.
    increase if snr is < 5db.

    """

    n, k = tG.shape
    if k != 17:
        raise ValueError("""coding matrix G must have 17 rows
                         (audio files are written in int16 which is
                         equivalent to uint17)""")

    length = audio_coded.shape[0]

    audio_decoded_bin = np.zeros(shape=(length, k), dtype=int)

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

    for j in range(length):

        decoded_vector = decodefunction(H, bits, nodes, audio_coded[j, :],
                                        snr, maxiter)
        if systematic:
            decoded_number = decoded_vector[:k]
        else:
            decoded_number = get_message(tG, decoded_vector)

        audio_decoded_bin[j, :] = decoded_number

    audio_decoded = bin2audio(audio_decoded_bin)

    return audio_decoded


def ber_audio(original_audio_bin, decoded_audio_bin):
    """
    Computes Bit-Error-Rate (BER) by comparing 2 binary audio arrays.
    The ratio of bit errors over total number of bits is returned.
    """
    if not original_audio_bin.shape == decoded_audio_bin.shape:
        raise ValueError("""Original and decoded audio files\'
                            shapes don\'t match !""")

    length, k = original_audio_bin.shape

    total_bits = np.prod(original_audio_bin.shape)

    errors_bits = abs(original_audio_bin-decoded_audio_bin).flatten().sum()

    ber = errors_bits / total_bits

    return(ber)
