=============================================
**Simulation of LDPC Codes & Applications**
=============================================
*version 0.6.2*

**Bug-fix release**

In Brief:
---------
- Generates coding and decoding matrices.
- Probabilistic decoding: Belief Propagation algorithm.
- Images transmission simulation (channel model: AGWN).
- Sound transmission simulation (channel model :AGWN).

**Image coding-decoding example:**

.. image:: https://media.giphy.com/media/l4KicsAauqIWjeFR6/giphy.gif
.. image:: https://media.giphy.com/media/l0COHC49bK6g7yIPm/giphy.gif



**Sound coding-decoding example:**

 `Sound Transmission <http://nbviewer.jupyter.org/github/janatiH/pyldpc/blob/master/Example-Sound.ipynb>`_

Tutorials:
----------

Jupyter notebooks:

- Users' Guide: 

1- `LDPC Coding-Decoding Simulation
<http://nbviewer.jupyter.org/github/janatiH/pyldpc/blob/master/pyLDPC-Tutorial-Basics.ipynb>`_

2- `Images Coding-DecodingTutorial <http://nbviewer.jupyter.org/github/janatiH/pyldpc/blob/master/pyLDPC-Tutorial-Images.ipynb?flush_cache=true>`_

3- `Sound Coding-DecodingTutorial <http://nbviewer.jupyter.org/github/janatiH/pyldpc/blob/master/pyLDPC-Tutorial-Sound.ipynb?flush_cache=true>`_

- For LDPC construction details:

1- `pyLDPC Construction <http://nbviewer.jupyter.org/github/janatiH/pyldpc/blob/master/pyLDPC-Presentation.ipynb>`_

2- `LDPC Images Functions Construction <http://nbviewer.jupyter.org/github/janatiH/pyldpc/blob/master/pyLDPC-Images-Construction.ipynb>`_
 
3- `LDPC Sound Functions Construction <http://nbviewer.jupyter.org/github/janatiH/pyldpc/blob/master/pyLDPC-Sound-Construction.ipynb>`_

version 0.6.2
-------------

 **Contains:**

1. Coding and decoding matrices Generators:
    - Regular parity-check matrix using Callager's method.
    - Coding Matrix G both non-systematic and systematic.
2. Coding function adding Additive White Gaussian Noise.
3. Decoding functions using Probabilistic Decoding (Belief propagation algorithm):
    - Default BP algorithm.
    - Full-log BP algorithm.
4. Images transmission sub-module:
    - Coding and Decoding Grayscale Images.
    - Coding and Decoding RGB Images.
    - BER: Bit Error Rate function.
5. Sound transmission sub-module:
    - Coding and Decoding audio files.
    - BER_audio: Bit Error Rate function.

 **What's new:**

- Sound sub-module.
- Bug in Importing ldpc_sound fixed.
- 25% faster calculations 

In the upcoming versions:
-------------------------

- Major Change: Use of large sparse matrices (spicy.sparse object).
- Optimized Parity Check Matrices (library).
- Text Transmission functions.

Contact:
--------
Please contact hicham.janati@ensae.fr for any bug encountered / any further information.