import numpy as np
import math


# def map_uint16_to_uint8(img, lower_bound=None, upper_bound=None):
#     '''
#     Map a 16-bit image trough a lookup table to convert it to 8-bit.

#     Returns
#     -------
#     numpy.ndarray
#     '''
#     if not(0 <= lower_bound < 2**16) and lower_bound is not None:
#         raise ValueError(
#             '"lower_bound" must be in the range [0, 65535]')
#     if not(0 <= upper_bound < 2**16) and upper_bound is not None:
#         raise ValueError(
#             '"upper_bound" must be in the range [0, 65535]')
#     if lower_bound is None:
#         lower_bound = np.min(img)
#     if upper_bound is None:
#         upper_bound = np.max(img)
#     if lower_bound >= upper_bound:
#         raise ValueError(
#             '"lower_bound" must be smaller than "upper_bound"')
#     lut = np.concatenate([
#         np.zeros(lower_bound, dtype=np.uint16),
#         np.linspace(0, 255, upper_bound - lower_bound).astype(np.uint16),
#         np.ones(2**16 - upper_bound, dtype=np.uint16) * 255
#     ])
#     return lut[img].astype(np.uint8)



class map_uint16_to_uint8(object):
    def __init__(self):
        super(map_uint16_to_uint8, self).__init__()

    def forward(self, img, lower_bound=None, upper_bound=None):

        if not(0 <= lower_bound < 2**16) and lower_bound is not None:
            raise ValueError(
            '"lower_bound" must be in the range [0, 65535]')
        if not(0 <= upper_bound < 2**16) and upper_bound is not None:
            raise ValueError(
                '"upper_bound" must be in the range [0, 65535]')
        if lower_bound is None:
            lower_bound = np.min(img)
        if upper_bound is None:
            upper_bound = np.max(img)
        if lower_bound >= upper_bound:
            raise ValueError(
                '"lower_bound" must be smaller than "upper_bound"')
        lut = np.concatenate([
            np.zeros(lower_bound, dtype=np.uint16),
            np.linspace(0, 255, upper_bound - lower_bound).astype(np.uint16),
            np.ones(2**16 - upper_bound, dtype=np.uint16) * 255
        ])
        return lut[img].astype(np.uint8)
