=============================================
**Simulation of LDPC Codes & Applications**
=============================================
*version 0.7.5*

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

 `Sound Transmission <http://nbviewer.jupyter.org/github/hichamjanati/pyldpc/blob/master/Example-Sound.ipynb>`_


Installation
------------

From pip::

    $ pip install --upgrade pyldpc


Tutorials:
----------

Jupyter notebooks:


*Many changes in tutorials in v.0.7.3*

- Users' Guide:

1- `LDPC Coding-Decoding Simulation
<http://nbviewer.jupyter.org/github/hichamjanati/pyldpc/blob/master/pyLDPC-Tutorial-Basics.ipynb?flush_cache=true>`_

2- `Images Coding-DecodingTutorial <http://nbviewer.jupyter.org/github/hichamjanati/pyldpc/blob/master/pyLDPC-Tutorial-Images.ipynb?flush_cache=true>`_

3- `Sound Coding-DecodingTutorial <http://nbviewer.jupyter.org/github/hichamjanati/pyldpc/blob/master/pyLDPC-Tutorial-Sound.ipynb?flush_cache=true>`_

4- `LDPC Matrices Construction Tutorial <http://nbviewer.jupyter.org/github/hichamjanati/pyldpc/blob/master/pyLDPC-Tutorial-Matrices.ipynb?flush_cache=true>`_

- For LDPC construction details:

1- `pyLDPC Construction(French) <http://nbviewer.jupyter.org/github/hichamjanati/pyldpc/blob/master/pyLDPC-Presentation.ipynb?flush_cache=true>`_

2- `LDPC Images Functions Construction <http://nbviewer.jupyter.org/github/hichamjanati/pyldpc/blob/master/pyLDPC-Images-Construction.ipynb?flush_cache=true>`_

3- `LDPC Sound Functions Construction <http://nbviewer.jupyter.org/github/hichamjanati/pyldpc/blob/master/pyLDPC-Sound-Construction.ipynb?flush_cache=true>`_

version 0.7.3
-------------

 **Contains:**

1. Coding and decoding matrices Generators:
    - Regular parity-check matrix using Callager's method.
    - Coding Matrix G both non-systematic and systematic.
2. Coding function adding Additive White Gaussian Noise.
3. Decoding functions using Probabilistic Decoding (Belief propagation algorithm):
    - Default and full-log BP algorithm.
4. Images transmission sub-module:
    - Coding and Decoding Grayscale and RGB Images.
5. Sound transmission sub-module:
    - Coding and Decoding audio files.
6. Compatibility numpy ndarrays <=> scipy sparse csr format.


 **What's new:**

- Python 2 compatibility


In the upcoming versions:
-------------------------

- Library of ready-to-use large matrices (csr).
- Text Transmission functions.

Contact:
--------
Please contact hicham.janati@ensae.fr for any bug encountered / any further information.
