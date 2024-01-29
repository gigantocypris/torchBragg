import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from create_fp_fdp_dat_file import full_path, read_dat_file

Mn_model=full_path("data_sherrell/MnO2_spliced.dat")
energy_vec, fp_vec, fdp_vec = read_dat_file(Mn_model)

energy_vec = np.array(energy_vec).astype(np.float64)
fp_vec = np.array(fp_vec).astype(np.float64)
fdp_vec = np.array(fdp_vec).astype(np.float64)


plt.figure(figsize=(100,10))
plt.plot(energy_vec, fp_vec, label="fp")
plt.plot(energy_vec, fdp_vec, label="fdp")
plt.legend()
plt.savefig("Mn_fp_fdp.png")

cs_fdp = CubicSpline(energy_vec, fdp_vec)

plt.figure(figsize=(100,10))
plt.plot(energy_vec, fdp_vec, 'r.', label="fdp")
plt.plot(energy_vec, cs_fdp(energy_vec), label="cs_fdp")
plt.legend()
plt.savefig("Mn_fdp_cs.png")


# Split curve from start to 6499.5eV, from 6499.5eV to 6599.5, 6599.5eV to end 
# 6550 eV is the bandedge of Mn
interval_0 = np.array([energy_vec[0], 6499.5])
interval_1 = np.array([6499.5, 6599.5]) 
interval_2 = np.array([6599.5, energy_vec[-1]])

# Fit interval_1 with a cubic spline
step_size = 1
energy_vec_bandwidth = np.arange(interval_1[0], interval_1[1] + step_size, step_size)
fdp_vec_bandwidth = cs_fdp(energy_vec_bandwidth)

cs_fdp_bandwidth = CubicSpline(energy_vec_bandwidth, fdp_vec_bandwidth)
coeff_bandwidth = cs_fdp_bandwidth.c

plt.figure()
plt.plot(energy_vec_bandwidth, fdp_vec_bandwidth, 'b.', label="fdp")
plt.plot(energy_vec_bandwidth, cs_fdp_bandwidth(energy_vec_bandwidth), 'r', label="cs_fdp")
plt.legend()
plt.savefig("Mn_fdp_cs_bandwidth.png")

# Fit interval_0 with a polynomial, enforce endpoint continuity


def func(x, shift, constant, a, b, c, d, e):
    return a*(x-shift)**3 + b*(x-shift)**2 + c*(x-shift) + d*(x)**(-1) - d*(shift)**(-1) + e*(x)**(-2) - e*(shift)**(-2) + constant

def convert_coeff(shift, constant, a, b, c, d, e):
    # convert to the form of f*x**(-2) + g*x**(-1) + h*x**0 + i*x**1 + j*x**2 + k*x**3
    f = e
    g = d
    h = -c*shift + a*shift**2 - b*shift**3 - d*(shift)**(-1) - e*(shift)**(-2) + constant
    i = c - a*2*shift + b*3*shift**2
    j = a - b*3*shift
    k = b
    return np.array([f,g,h,i,j,k])

"""
coeff for x**3
b*x**3

coeff for x**2
a*x**2 - b*3*shift*x**2

coeff for x
c*x - a*2*shift*x + b*3*shift**2*x

coeff for x**(0)
-c*shift + a*shift**2 - b*shift**3 - d*(shift)**(-1) - e*(shift)**(-2) + constant

coeff for x**(-1)
d*(x)**(-1)

coeff for x**(-2)
e*(x)**(-2)

"""

energy_vec_0 = energy_vec[energy_vec <= interval_0[1]]
fdp_vec_0 = fdp_vec[energy_vec <= interval_0[1]]
if energy_vec_0[-1] != interval_0[1]:
    energy_vec_0 = np.append(energy_vec_0, interval_0[1])
    fdp_vec_0 = np.append(fdp_vec_0, cs_fdp_bandwidth(interval_0[1]))

shift_0 = energy_vec_0[-1]
constant_0 = fdp_vec_0[-1]

func_fixed_pt_0 = lambda x, a, b, c, d, e: func(x, shift_0, constant_0, a, b, c, d, e)

popt_0, pcov_0 = curve_fit(func_fixed_pt_0, energy_vec_0, fdp_vec_0)   

plt.figure()
plt.plot(energy_vec_0, fdp_vec_0, 'b.', label="fdp")
plt.plot(energy_vec_0, func_fixed_pt_0(energy_vec_0, *popt_0), 'r', label="fit")
plt.legend()
plt.savefig("Mn_fdp_fit_0.png")

# Fit interval_1 with a polynomial, enforce endpoint continuity

energy_vec_2 = energy_vec[energy_vec >= interval_2[0]]
fdp_vec_2 = fdp_vec[energy_vec >= interval_2[0]]
if energy_vec_2[0] != interval_2[0]:
    energy_vec_2 = np.append(interval_2[0], energy_vec_2)
    fdp_vec_2 = np.append(cs_fdp_bandwidth(interval_2[0]), fdp_vec_2)

shift_2 = energy_vec_2[0]
constant_2 = fdp_vec_2[0]

func_fixed_pt_2 = lambda x, a, b, c, d, e: func(x, shift_2, constant_2, a, b, c, d, e)

popt_2, pcov_2 = curve_fit(func_fixed_pt_2, energy_vec_2, fdp_vec_2)   

plt.figure()
plt.plot(energy_vec_2, fdp_vec_2, 'b.', label="fdp")
plt.plot(energy_vec_2, func_fixed_pt_2(energy_vec_2, *popt_2), 'r', label="fit")
plt.legend()
plt.savefig("Mn_fdp_fit_2.png")

# Plot the entire fit

energy_vec_full = np.concatenate((energy_vec_0, energy_vec_bandwidth, energy_vec_2), axis=0)
fdp_vec_full = np.concatenate((func_fixed_pt_0(energy_vec_0, *popt_0), cs_fdp_bandwidth(energy_vec_bandwidth), func_fixed_pt_2(energy_vec_2, *popt_2)), axis=0)

plt.figure(figsize=(200,10))
plt.plot(energy_vec, fdp_vec, 'r.', label="original")
plt.plot(energy_vec_full, fdp_vec_full, 'b', label="fdp fit")
plt.legend()
plt.savefig("Mn_fdp_full.png")

# Convert fdp to fp using the Kramers-Kronig relation

def fdp_fp_easy_integral(energy, energy_start, energy_end, coeff, powers):
    """
    Get integral at energy for the term with the x+E denominator
    """

    # Check that powers is sorted in ascending order
    assert all(powers[i] <= powers[i+1] for i in range(len(powers)-1))

    integral = 0
    for ind,n in enumerate(powers):
        coeff_i = coeff[ind]
        if n >= 0:
            for k in range(1,n+2):
                integral += coeff_i*(((-energy)**(n-k+1))/k)*(energy_start**k - energy_end**k)
        integral += coeff_i*((-energy)**(n+1))*np.log((energy_end + energy)/(energy_start+energy))
        if n <= -2:
            integral += -coeff_i*((-energy)**(n+1))*np.log(energy_end/energy_start)
        if n <= -3:
            for k in range(n+2,0):
                integral += coeff_i*(((-energy)**(n-k+1))/(-k))*(energy_end**k - energy_start**k)
    return integral


def fdp_fp_hard_integral(energy, energy_start, energy_end, coeff, powers):
    """
    Get integral at energy for the term with the x-E denominator
    """

    # Check that powers is sorted in ascending order
    assert all(powers[i] <= powers[i+1] for i in range(len(powers)-1))

    integral = 0
    for ind,n in enumerate(powers):
        coeff_i = coeff[ind]
        if n >= 0:
            for k in range(1,n+2):
                integral += coeff_i*((energy**(n-k+1))/k)*(energy_start**k - energy_end**k)
        integral += coeff_i*(energy**(n+1))*np.log((energy_end - energy)/(energy_start - energy)) ## Problem term here
        if n <= -2:
            integral += -coeff_i*(energy**(n+1))*np.log(energy_end/energy_start)
        if n <= -3:
            for k in range(n+2,0):
                integral += coeff_i*((energy**(n-k+1))/(-k))*(energy_end**k - energy_start**k)
    return integral

def fdp_fp_integrate(energy, intervals_mat, coeff_mat, powers_mat):
    """
    Find fp from fdp using the Kramers-Kronig relation at energy
    energy cannot be at the endpoints of any interval
    powers must be sorted in ascending order

    Use the analytical expressions for the integrals from Watts (2014)
    Notation in Watts (2014):
        E is the energy where we want to find fp (energy in this case)
        Degree of polynomial starts at M, M can be negative
        Degree of polynomial ends at N, assume N is positive
        jth interval
    """

    for interval in intervals_mat:
        pass


# Create intervals_mat, coeff_mat, powers_mat

# interval_1 is broken up into intervals depending on step size
interval_1_starts = np.arange(interval_1[0], interval_1[1], step_size)
interval_1_ends = np.arange(interval_1[0]+step_size, interval_1[1] + step_size, step_size)
interval_1_all = np.array([interval_1_starts, interval_1_ends]).T

intervals_mat = np.concatenate([np.expand_dims(interval_0,axis=0), interval_1_all, np.expand_dims(interval_2, axis=0)],axis=0)

# XXX STOPPED HERE
breakpoint()
powers_mat = np.array([])
coeff_mat

