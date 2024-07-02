"""
A PyTorch implementation of a natural cubic spline fit is in cubic_spline_torch.py.
This test checks whether the coefficients from the PyTorch implementation match the scipy CubicSpline implementation.

Usage:
libtbx.python $MODULES/torchBragg/tests/tst_cubic_spline_torch.py
"""

import numpy as np
import torch
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from torchBragg.kramers_kronig.cubic_spline_torch import natural_cubic_spline_coeffs_without_missing_values

energy_vec = torch.linspace(0, 6, 10)
fdp_vec = energy_vec.sin()


cs_fdp = CubicSpline(energy_vec, fdp_vec, bc_type='natural')
coeff_bandwidth = cs_fdp.c

a, b, c, d = natural_cubic_spline_coeffs_without_missing_values(energy_vec, fdp_vec)
coeff_check = torch.stack([d,c,b,a])

np.testing.assert_allclose(coeff_bandwidth, coeff_check, rtol=1e-5, atol=0)