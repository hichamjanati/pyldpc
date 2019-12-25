import numpy as np
from . import utils


def gray2bin(img):
    """Convert a GrayScale Image to a binary array."""
    if not len(img.shape) == 2:
        raise ValueError("""{} must have 2 dimensions.
                         Make sure it\'s a grayscale image.""")

    height, width = img.shape

    img_bin = np.zeros(shape=(height, width, 8), dtype=int)

    for i in range(height):
        for j in range(width):
            img_bin[i, j, :] = utils.int2bitarray(img[i, j], 8)

    return img_bin


def bin2gray(img_bin):
    """Convert a binary Image to a grayscale image."""
    height, width, k = img_bin.shape
    img_grayscale = np.zeros(shape=(height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            img_grayscale[i, j] = utils.bitarray2int(img_bin[i, j, :])

    return img_grayscale


def rgb2bin(img):
    """Convert an RGB Image to a binary array."""
    height, width, depth = img.shape

    if not depth == 3:
        raise ValueError("""{}\'s 3rd dimension must be equal to 3 (RGB).
                             Make sure it\'s an RGB image.""")

    img_bin = np.zeros(shape=(height, width, 24), dtype=int)

    for i in range(height):
        for j in range(width):
            r = utils.int2bitarray(img[i, j, 0], 8)
            g = utils.int2bitarray(img[i, j, 1], 8)
            b = utils.int2bitarray(img[i, j, 2], 8)

            img_bin[i, j, :] = np.concatenate((r, g, b))

    return img_bin


def bin2rgb(img_bin):
    """Convert a binary image to RGB."""
    height, width, depth = img_bin.shape
    img_rgb = np.zeros(shape=(height, width, 3), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            r = utils.bitarray2int(img_bin[i, j, :8])
            g = utils.bitarray2int(img_bin[i, j, 8:16])
            b = utils.bitarray2int(img_bin[i, j, 16:])

            img_rgb[i, j] = np.array([r, g, b], dtype=np.uint8)

    return img_rgb
