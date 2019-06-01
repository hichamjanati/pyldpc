# import numpy as np
#
# from pyldpc import (make_ldpc, binaryproduct, encode_random_message, decode,
#                     get_message, encode, decoder)
# import pytest
# from itertools import product
#
#
# # def test_decoding_random(systematic, log, sparse):
# if 1:
#     n = 1000
#     d_v = 4
#     d_c = 5
#     seed = 0
#
#     H, G = make_ldpc(n, d_v, d_c, seed=seed, systematic=True,
#                      sparse=False)
#     assert not binaryproduct(H, G).any()
#     n, k = G.shape
#     snr = 100
#
#     v, y = encode_random_message(G, snr, seed)
#
#     d = decoder.decode_bp(H, y, snr, maxiter=50)
#     x = get_message(G, d)
#
#     assert abs(v - x).sum() == 0
