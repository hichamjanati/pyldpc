.. _changelog:

==========
Change Log
==========


Version 0.7.8
--------------

- Fix a major bug in creation of the LDPC matrix H with Gallager's algorithm.

- 10x speed gain in decoding with numba

- Make coding and decoding parallelizable by stacking multiple messages.

- Make images and sound modules compact via parallel coding and decoding.

- Move documentation from readthedocs to github-pages + add example on images.
