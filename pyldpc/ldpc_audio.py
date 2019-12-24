import numpy as np
import warnings

from .utils_audio import bin2audio
from .encoder import encode
from .decoder import _decode_logbp_ext, get_message
from .utils import bitsandnodes


def encode_audio(tG, audio_bin, snr, seed=None):
    """Encode a binary audio file.

    Parameters
    ----------
    tG: array (n, 17). Coding matrix. `k` is the number of bits to be coded.
        k must be equal to 17 for audio files. `n` is the length of the
        codewords.
    audio_bin: array (length, 17). Binary audio.
    snr : float. Signal to noise ratio of the channel.
    seed: int. random state initialization.

    Returns
    -------
    coded_audio: array (length n) audio in the codeword space
    noisy_audio: array (length, 17) visualization of the audio data.

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


def decode_audio(tG, H, audio_coded, snr, maxiter=1000):
    """Decode a received noisy audio file in the codeword.

    Parameters
    ----------
    tG: array (n, k) coding matrix G
    H: array (m, n) decoding matrix H
    audio_coded: array (length n) audio in the codeword space
    snr: float. signal to noise ratio assumed of the channel.
    maxiter: int. Max number of BP iterations to perform.

    Returns
    -------
    audio_decoded: array (length,) original audio.

    """
    n, k = tG.shape
    if k != 17:
        raise ValueError("""coding matrix G must have 17 rows
                         (audio files are written in int16 which is
                         equivalent to uint17)""")

    length = audio_coded.shape[0]

    audio_decoded_bin = np.zeros(shape=(length, k), dtype=int)

    decodefunction = _decode_logbp_ext
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
    """Compute Bit-Error-Rate (BER) by comparing 2 binary audio arrays."""
    if not original_audio_bin.shape == decoded_audio_bin.shape:
        raise ValueError("""Original and decoded audio files\'
                            shapes don\'t match !""")

    length, k = original_audio_bin.shape

    total_bits = np.prod(original_audio_bin.shape)

    errors_bits = abs(original_audio_bin-decoded_audio_bin).flatten().sum()

    ber = errors_bits / total_bits

    return(ber)
