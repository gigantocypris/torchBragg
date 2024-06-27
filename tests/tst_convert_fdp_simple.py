import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tst_convert_fdp_helper import fdp_fp_integrate
np.seterr(all='raise')

step_size = .1
energy_max = 10.
energy_vec = np.arange(step_size,energy_max,step_size)
fdp_vec = 1/energy_vec
fp_vec = 1/energy_vec/np.pi*np.log((energy_max+energy_vec)/(energy_max-energy_vec))
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
plt.savefig("sine_fit.png")

# Convert fdp to fp

intervals_mat = np.array([[0, energy_max]])
coeff_mat = np.array([popt_0])
powers_mat = np.array([powers])

fp_calculated = []
for energy in energy_vec[1:-1]:
    fp = fdp_fp_integrate(energy, intervals_mat, coeff_mat, powers_mat, relativistic_correction)
    fp_calculated.append(fp)

plt.figure()
plt.plot(energy_vec[1:-1], fp_vec[1:-1], 'r.', label="original")
plt.plot(energy_vec[1:-1], fp_calculated, 'b', label="calculated")
plt.legend()
plt.savefig("sine_fp_calculated.png")
