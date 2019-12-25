import numpy as np
import warnings

from .utils_audio import bin2audio
from .encoder import encode
from .decoder import get_message, decode


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
    length, depth = audio_bin.shape
    if depth != 17:
        raise ValueError("The last dimension of `audio_bin` must be 17."
                         "Got %s. See `pyldpc.utils.audio2bin`." % depth)

    audio_bin = audio_bin.flatten()
    n_bits_total = audio_bin.size
    n_blocks = n_bits_total // k
    residual = n_bits_total % k
    if residual:
        n_blocks += 1
    resized_audio = np.zeros(k * n_blocks)
    resized_audio[:n_bits_total] = audio_bin

    codeword = encode(tG, resized_audio.reshape(k, n_blocks), snr, seed)
    noisy_audio = (codeword.flatten()[:n_bits_total] < 0).astype(int)
    noisy_audio = noisy_audio.reshape(length, depth)
    noisy_audio = bin2audio(noisy_audio)

    return codeword, noisy_audio


def decode_audio(tG, H, codeword, snr, audio_shape, maxiter=1000):
    """Decode a received noisy audio file in the codeword.

    Parameters
    ----------
    tG: array (n, k) coding matrix G
    H: array (m, n) decoding matrix H
    audio_coded: array (length n) audio in the codeword space
    snr: float. signal to noise ratio assumed of the channel.
    audio_shape: tuple (2,). Shape of original audio data.
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

    _, n_blocks = codeword.shape

    # decodefunction = _decode_logbp_ext
    systematic = True

    if not (tG[:k, :] == np.identity(k)).all():
        warnings.warn("""In LDPC applications, using systematic coding matrix
                         G is highly recommanded to speed up decode.""")
        systematic = False

    codeword_solution = decode(H, codeword, snr, maxiter)
    if systematic:
        decoded = codeword_solution[:k, :]
    else:
        decoded = np.array([get_message(tG, codeword_solution[:, i])
                           for i in range(n_blocks)]).T
    decoded = decoded.flatten()[:np.prod(audio_shape)]
    decoded = decoded.reshape(*audio_shape)

    audio_decoded = bin2audio(decoded)

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
