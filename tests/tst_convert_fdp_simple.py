"""
For fdp = 1/energy, we can evaluate the Kramers-Kronig integral analytically to get the exact fp expression.
Resources: 
https://en.wikipedia.org/wiki/Kramers%E2%80%93Kronig_relations
https://www.wolframalpha.com/

We compare the results of our function fdp_fp_integrate.

Usage:
libtbx.python $MODULES/torchBragg/tests/tst_convert_fdp_simple.py
"""


import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from torchBragg.kramers_kronig.convert_fdp_helper import fdp_fp_integrate
np.seterr(all='raise')

step_size = .1
energy_max = 5.
energy_vec = np.arange(step_size,energy_max,step_size) # cannot evaluate at the exact interval endpoints
fdp_vec = 1/energy_vec
fp_vec = 1/energy_vec/np.pi*(np.log((energy_max+energy_vec)/(energy_max-energy_vec)))
relativistic_correction = 0

# Fit with a polynomial
def func(x, powers, coeffs):
    y=np.zeros_like(x)
    for p, c in zip(powers,coeffs):
        y += c*(x**p)
    return y

powers = np.array([-1])
func_fix_powers = lambda x,a: func(x, powers, [a])

popt_0, pcov_0 = curve_fit(func_fix_powers, energy_vec, fdp_vec)   

# Plot the fit

plt.figure()
plt.plot(energy_vec, fdp_vec, 'r.', label="original")
plt.plot(energy_vec, func_fix_powers(energy_vec, *popt_0), 'b', label="fit")
plt.legend()
plt.savefig("tst_fdp_fit.png")


# Convert fdp to fp

intervals_mat = torch.tensor([[0, energy_max]])
coeff_mat = torch.tensor([popt_0])
powers_mat = torch.tensor([powers])

fp_calculated = []
for energy in energy_vec:
    fp = fdp_fp_integrate(energy, intervals_mat, coeff_mat, powers_mat, relativistic_correction)
    fp_calculated.append(fp)

plt.figure()
plt.plot(energy_vec, fp_vec, 'r.', label="original")
plt.plot(energy_vec, fp_calculated, 'b', label="calculated")
plt.legend()
plt.savefig("tst_fp_calculated.png")

np.testing.assert_allclose(fp_vec, fp_calculated, rtol=1e-5, atol=0)