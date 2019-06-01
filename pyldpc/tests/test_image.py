# import numpy as np
#
# from pyldpc import (make_ldpc, binaryproduct, ldpc_images)
# import pytest
# from itertools import product
#
#
# @pytest.mark.parametrize("systematic, log, sparse",
#                          product([False, True], [True], [False, True]))
# def test_image_gray(systematic, log, sparse):
#     n = 60
#     d_v = 4
#     d_c = 5
#     seed = 0
#     rnd = np.random.RandomState(seed)
#     H, G = make_ldpc(n, d_v, d_c, seed=seed, systematic=systematic)
#     assert not binaryproduct(H, G).any()
#
#     n, k = G.shape
#     snr = 100
#
#     img = rnd.randint(255, size=(3, 3))
#
#     y = encode(G, v, snr, seed)
#
#     d = decode(H, y, snr, maxiter=10, log=log)
#     x = get_message(G, d)
#
#     assert abs(v - x).sum() == 0
#
#
# # @pytest.mark.parametrize("systematic, log, sparse",
# #                          product([False, True], [True], [False, True]))
# # def test_image_rgb(systematic, log, sparse):
# #     n = 15
# #     d_v = 4
# #     d_c = 5
# #     seed = 0
# #
# #     H, G = make_ldpc(n, d_v, d_c, seed=seed, systematic=systematic)
# #     assert not binaryproduct(H, G).any()
# #
# #     n, k = G.shape
# #     snr = 100
# #
# #     v = np.arange(k) % 2
# #     y = encode(G, v, snr, seed)
# #
# #     d = decode(H, y, snr, maxiter=10, log=log)
# #     x = get_message(G, d)
# #
# #     assert abs(v - x).sum() == 0
