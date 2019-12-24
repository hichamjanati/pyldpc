import numpy as np

from .utils import int2bitarray, bitarray2int


def audio2bin(audio_array):
    """Convert an audio_array to a 17-bits binary array."""
    # Keep the first channel of the audio file only:
    if len(audio_array.shape) > 1:
        audio = audio_array[:, 0]
    else:
        audio = audio_array

    length = audio.size

    # Translate audio by 2^15 so as to make its dtype unsigned.
    audio = audio + 2 ** 15

    audio_bin = np.zeros(shape=(length, 17), dtype=int)
    for i in range(length):
        audio_bin[i, :] = int2bitarray(audio[i], 17)

    return audio_bin


def bin2audio(audio_bin):
    """Convert a 17-bits binary array to an audio array."""
    length = audio_bin.shape[0]

    audio = np.zeros(length, dtype=int)

    for i in range(length):
        audio[i] = bitarray2int(audio_bin[i, :])

    # Translate audio by - 2^15 so as to make its dtype signed int16.

    audio = audio - 2 ** 15

    return audio.astype(np.int16)
