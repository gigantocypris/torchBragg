import numpy as np
import matplotlib.pyplot as plt
np.seterr(all='raise')

def func(x, shift, constant, a, b, c, d, e):
    return a*(x-shift)**3 + b*(x-shift)**2 + c*(x-shift) + d*(x)**(-1) - d*(shift)**(-1) + e*(x)**(-2) - e*(shift)**(-2) + constant

def convert_coeff(shift, constant, a, b, c, d, e):
    # convert to the form of f*x**(-2) + g*x**(-1) + h*x**0 + i*x**1 + j*x**2 + k*x**3
    f = e
    g = d
    h = -c*shift + b*shift**2 - a*shift**3 - d*(shift)**(-1) - e*(shift)**(-2) + constant
    i = c - b*2*shift + a*3*shift**2
    j = b - a*3*shift
    k = a
    return np.array([f,g,h,i,j,k])

def func_converted_coeff(x, power_vec, coeff_vec):
    y = np.zeros_like(x)
    for p, c in zip(power_vec,coeff_vec):
            y += c*(x**p)
    return y

def find_fdp(energy, powers_mat, coeff_mat, intervals_mat):
    # find what interval energy is in
    interval_ind = intervals_mat - energy # the interval that energy is in has a change from negative to positive
    interval_ind[interval_ind<=0] = 0 # equal sign should never be triggered!
    interval_ind[interval_ind>0] = 1
    interval_ind = interval_ind.astype(bool)
    interval_ind = np.sum(interval_ind, axis=1)*(1-np.prod(interval_ind, axis=1)) # XOR
    interval_ind = np.where(interval_ind)[0][0]

    power = powers_mat[interval_ind]
    coeff = coeff_mat[interval_ind]
    interval = intervals_mat[interval_ind]
    fdp_fit = func_converted_coeff(energy, power, coeff)
    return fdp_fit


def plot_fit(powers_mat, coeff_mat, intervals_mat, energy_vec, fdp_vec):
    energy_points_vec = []
    fdp_fit_vec = []
    for i in range(len(powers_mat)):
        power = powers_mat[i]
        coeff = coeff_mat[i]
        interval = intervals_mat[i]
        energy_points = np.linspace(interval[0], interval[1], 100)
        fdp_fit = func_converted_coeff(energy_points, power, coeff)
        energy_points_vec.append(energy_points)
        fdp_fit_vec.append(fdp_fit)

    energy_points_vec = np.concatenate(energy_points_vec, axis=0)
    fdp_fit_vec = np.concatenate(fdp_fit_vec, axis=0)

    plt.figure(figsize=(20,10))
    plt.plot(energy_vec, fdp_vec, 'r.', label="original")
    plt.plot(energy_points_vec, fdp_fit_vec, 'b', label="fdp fit")
    plt.xlim([7000,7200])
    plt.legend()
    plt.savefig("Mn_fdp_fit_check.png")

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
                integral += coeff_i*(((-energy)**(n-k+1))/k)*(energy_end**k - energy_start**k)
        integral += coeff_i*((-energy)**(n+1))*np.log(np.abs((energy_end + energy)/(energy_start+energy)))
        if n <= -2:
            integral += -coeff_i*((-energy)**(n+1))*np.log(np.abs(energy_end/energy_start))
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
                integral += coeff_i*((energy**(n-k+1))/k)*(energy_end**k - energy_start**k)
        try:
            integral += coeff_i*(energy**(n+1))*np.log(np.abs((energy_end - energy)/(energy_start - energy))) ## Problem term here
        except FloatingPointError as rw:
            print("FloatingPointError:", rw)
            print("Check that energy is not at the endpoints of any interval")
        if n <= -2:
            integral += -coeff_i*(energy**(n+1))*np.log(np.abs(energy_end/energy_start))
        if n <= -3:
            for k in range(n+2,0):
                integral += coeff_i*((energy**(n-k+1))/(-k))*(energy_end**k - energy_start**k)
    return integral

def fdp_fp_integrate(energy, intervals_mat, coeff_mat, powers_mat, relativistic_correction):
    """
    Find fp from fdp using the Kramers-Kronig relation at energy
    energy cannot be at the endpoints of any interval!!
    powers must be sorted in ascending order

    Use the analytical expressions for the integrals from Watts (2014)
    Notation in Watts (2014):
        E is the energy where we want to find fp (energy in this case)
        Degree of polynomial starts at M, M can be negative
        Degree of polynomial ends at N, assume N is positive
        jth interval
    """
    fp = 0
    for ind, interval in enumerate(intervals_mat):
        energy_start = interval[0]
        energy_end = interval[1]
        coeff = coeff_mat[ind]
        powers = powers_mat[ind]
        fp += 1/(np.pi*energy)*fdp_fp_easy_integral(energy, energy_start, energy_end, coeff, powers)
        fp += -1/(np.pi*energy)*fdp_fp_hard_integral(energy, energy_start, energy_end, coeff, powers)
    return fp + relativistic_correction