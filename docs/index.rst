.. project-template documentation master file, created by
   sphinx-quickstart on Mon Jan 18 14:44:12 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyldpc documentation
===============================

*version 0.7.8*

Description:
------------
- Simulation of regular LDPC codes.
- Probabilistic decoding: Belief Propagation algorithm for gaussian white noise transmission.
- Simulation application to image and audio data.

**Image coding-decoding example:**

.. image:: https://media.giphy.com/media/l4KicsAauqIWjeFR6/giphy.gif
.. image:: https://media.giphy.com/media/l0COHC49bK6g7yIPm/giphy.gif


**Sound coding-decoding example:**

 `Sound Transmission <http://nbviewer.jupyter.org/github/hichamjanati/pyldpc-tutos/blob/master/Example-Sound.ipynb>`_


Installation
------------

If you already have a working Python environment (Anaconda for e.g):

::

    pip install --upgrade pyldpc

Otherwise, we recommend creating this minimal `conda env <https://raw.githubusercontent.com/hichamjanati/pyldpc/master/environment.yml>`_

::

    conda env create --file environment.yml
    conda activate pyldpc-env
    pip install -U pyldpc

Example
-------

.. code:: python

    >>> import numpy as np
    >>> from pyldpc import make_ldpc, encode, decode, get_message
    >>> n = 15
    >>> d_v = 4
    >>> d_c = 5
    >>> snr = 100
    >>> H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True)
    >>> k = G.shape[1]
    >>> v = np.random.randint(2, size=k)
    >>> y = encode(G, v, snr)
    >>> d = decode(H, y, snr)
    >>> x = get_message(G, d)
    >>> assert abs(x - v).sum() == 0

Contact:
--------
Please create an issue on `Github <https://github.com/hichamjanati/pyldpc>`_ for any bug encountered / any further information.

Dependencies
============

All dependencies are in ``./environment.yml``

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Documentation

   api

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Tutorial - Examples

   auto_examples/index

`Examples <auto_examples/index.html>`_
--------------------------------------

A set of examples: `User Guide <auto_examples/index.html>`_.
