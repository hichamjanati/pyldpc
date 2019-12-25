.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_plot_image_transmission.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_plot_image_transmission.py:


Coding - Decoding simulation of an image
========================================

This example shows a simulation of the transmission of an image as a
binary message through a gaussian white noise channel with an LDPC coding and
decoding system.


.. code-block:: default



    # Author: Hicham Janati (hicham.janati@inria.fr)
    #
    # License: BSD (3-clause)


    import numpy as np
    from pyldpc import make_ldpc, ldpc_images
    from pyldpc.utils_img import gray2bin, rgb2bin
    from matplotlib import pyplot as plt
    from PIL import Image

    from time import time







Let's see the image we are going to be working with


.. code-block:: default

    eye = Image.open("data/eye.png")
    # convert it to grayscale and keep one channel
    eye = np.asarray(eye.convert('LA'))[:, :, 0]

    # Convert it to a binary matrix
    eye_bin = gray2bin(eye)
    print("Eye shape: (%s, %s)" % eye.shape)
    print("Binary Eye shape: (%s, %s, %s)" % eye_bin.shape)

    n = 200
    d_v = 3
    d_c = 4
    seed = 42





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Eye shape: (128, 128)
    Binary Eye shape: (128, 128, 8)



First we create a small LDPC code i.e a pair of decoding and coding matrices
H and G. H is a regular parity-check matrix with d_v ones per row
and d_c ones per column


.. code-block:: default


    H, G = make_ldpc(n, d_v, d_c, seed=seed, systematic=True, sparse=True)







Now we simulate the transmission with Gaussian white noise
and recover the original image via belief-propagation.


.. code-block:: default


    snr = 8
    eye_coded, eye_noisy = ldpc_images.encode_img(G, eye_bin, snr)

    print("Coded eye shape", eye_coded.shape)

    t = time()
    eye_decoded = ldpc_images.decode_img(G, H, eye_coded, snr, eye_bin.shape)
    t = time() - t
    print("Eye | Decoding time: ", t)
    error_decoded_eye = abs(eye - eye_decoded).mean()
    error_noisy_eye = abs(eye_noisy - eye).mean()






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Coded eye shape (200, 2521)
    Eye | Decoding time:  0.660240888595581



With RGB images, we proceed similarly


.. code-block:: default

    print("\n\n")
    tiger = np.asarray(Image.open("data/tiger.jpg"))
    # Convert it to a binary matrix
    tiger_bin = rgb2bin(tiger)
    print("Tiger shape: (%s, %s, %s)" % tiger.shape)
    print("Tiger Binary shape: (%s, %s, %s)" % tiger_bin.shape)


    tiger_coded, tiger_noisy = ldpc_images.encode_img(G, tiger_bin, snr)

    print("Coded Tiger shape", tiger_coded.shape)

    t = time()
    tiger_decoded = ldpc_images.decode_img(G, H, tiger_coded, snr, tiger_bin.shape)
    t = time() - t
    print("Tiger | Decoding time: ", t)

    error_decoded_tiger = abs(tiger - tiger_decoded).mean()
    error_noisy_tiger = abs(tiger_noisy - tiger).mean()


    titles_eye = ["Original", "Noisy | Err = %.2f %%" % error_noisy_eye,
                  "Decoded | Err = %.2f %%" % error_decoded_eye]
    titles_tiger = ["Original", "Noisy | Err = %.2f %%" % error_noisy_tiger,
                    "Decoded | Err = %.2f %%" % error_decoded_tiger]
    all_imgs = [[eye, eye_noisy, eye_decoded], [tiger, tiger_noisy, tiger_decoded]]
    f, axes = plt.subplots(2, 3, figsize=(18, 12))
    for ax_row, titles, img_list, cmap in zip(axes, [titles_eye, titles_tiger],
                                              all_imgs, ["gray", None]):
        for ax, data, title in zip(ax_row, img_list, titles):
            ax.imshow(data, cmap=cmap)
            ax.set_title(title, fontsize=20)
            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout()
    plt.show()



.. image:: /auto_examples/images/sphx_glr_plot_image_transmission_001.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none




    Tiger shape: (128, 128, 3)
    Tiger Binary shape: (128, 128, 24)
    Coded Tiger shape (200, 7562)
    Tiger | Decoding time:  1.9829249382019043
    /Users/hichamjanati/Documents/github/pyldpc/examples/plot_image_transmission.py:101: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
      plt.show()




.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  4.647 seconds)

**Estimated memory usage:**  159 MB


.. _sphx_glr_download_auto_examples_plot_image_transmission.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_image_transmission.py <plot_image_transmission.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_image_transmission.ipynb <plot_image_transmission.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
