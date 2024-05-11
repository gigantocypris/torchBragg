import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
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

def create_figures(energy_vec, fp_vec, fdp_vec, cs_fdp, energy_vec_bandwidth,
                   fdp_vec_bandwidth, cs_fdp_bandwidth, energy_vec_0, fdp_vec_0,
                   func_fixed_pt_0, popt_0, energy_vec_2, fdp_vec_2, func_fixed_pt_2,
                   popt_2, energy_vec_full, fdp_vec_full, shift_0, constant_0, fp_calculate_bandwidth,
                   energy_vec_bandwidth_final,
                   prefix="Mn"):
    # plt.figure(figsize=(20,10))
    plt.figure()
    plt.plot(energy_vec, fp_vec, 'r', label="fp")
    plt.plot(energy_vec, fdp_vec, 'b', label="fdp")
    plt.legend()
    plt.savefig(prefix + "_fp_fdp.png")

    plt.figure()
    plt.plot(energy_vec, fdp_vec, 'b', label="fdp")
    # plt.xlim([2000,10000])
    plt.legend()
    # vertical_line = np.arange(min(fdp_vec),max(fdp_vec))
    # plt.plot(energy_vec_bandwidth[0]*np.ones_like(vertical_line), vertical_line, 'black')
    # plt.plot(energy_vec_bandwidth[-1]*np.ones_like(vertical_line), vertical_line, 'black')
    plt.savefig(prefix + "_fdp.png")

    plt.figure(figsize=(20,10))
    plt.plot(energy_vec, fdp_vec, 'r.', label="fdp")
    plt.plot(energy_vec, cs_fdp(energy_vec), label="cs_fdp")
    plt.legend()
    plt.savefig(prefix + "_fdp_cs.png")

    plt.figure()
    plt.xlim([energy_vec_bandwidth[0],energy_vec_bandwidth[-1]])
    plt.plot(energy_vec, fdp_vec, 'b', label="fdp")
    plt.plot(energy_vec_bandwidth, cs_fdp_bandwidth(energy_vec_bandwidth), 'r', label="cs_fdp")
    plt.legend()
    plt.savefig(prefix + "_fdp_cs_bandwidth.png")

    plt.figure()
    plt.plot(energy_vec_0, fdp_vec_0, 'b.', label="fdp")
    plt.plot(energy_vec_0, func_fixed_pt_0(energy_vec_0, *popt_0), 'r', label="fit")
    plt.legend()
    plt.savefig(prefix + "_fdp_fit_0.png")

    plt.figure()
    plt.plot(energy_vec_2, fdp_vec_2, 'b.', label="fdp")
    plt.plot(energy_vec_2, func_fixed_pt_2(energy_vec_2, *popt_2), 'r', label="fit")
    plt.legend()
    plt.savefig(prefix + "_fdp_fit_2.png")

    plt.figure(figsize=(20,10))
    plt.plot(energy_vec, fdp_vec, 'r.', label="original")
    plt.plot(energy_vec_full, fdp_vec_full, 'b', label="fdp fit")
    plt.legend()
    plt.savefig(prefix + "_fdp_full.png")

    plt.figure()
    plt.plot(energy_vec_0, fdp_vec_0, 'b.', label="fdp")
    plt.plot(energy_vec_0, func_converted_coeff(energy_vec_0, np.array([-2,-1,0,1,2,3]), convert_coeff(shift_0, constant_0, *popt_0)), 'g', label="fit2")
    plt.legend()
    plt.savefig(prefix + "_fdp_fit2_0.png")

    plt.figure()
    plt.plot(energy_vec, fp_vec, 'r.', label="original")
    plt.plot(energy_vec_bandwidth_final, fp_calculate_bandwidth, 'b', label="calculated")
    plt.legend()
    plt.savefig(prefix + "_fp_calculated.png")

    plt.figure()
    plt.plot(energy_vec, fp_vec, 'r.', label="original")
    plt.plot(energy_vec_bandwidth_final, fp_calculate_bandwidth, 'b', label="calculated")
    plt.xlim([energy_vec_bandwidth[0],energy_vec_bandwidth[-1]])
    plt.legend()
    plt.savefig(prefix + "_fp_calculated_bandwidth.png")

def convert_fdp_to_fp(energy_vec, fdp_vec, bandedge, relativistic_correction):

    # Fit the entire curve with a cubic spline, will use this for resampling points
    cs_fdp = CubicSpline(energy_vec, fdp_vec)

    # Split the curve into 3 parts, before bandedge, around bandedge, after bandedge
    interval_0 = np.array([energy_vec[0]-100, bandedge - 50.5])
    interval_1 = np.array([bandedge - 50.5, bandedge + 49.5]) 
    interval_2 = np.array([bandedge + 49.5, energy_vec[-1]+10000])

    # Fit interval_1 with its own a cubic spline
    step_size = 1
    energy_vec_bandwidth = np.arange(interval_1[0], interval_1[1] + step_size, step_size)
    fdp_vec_bandwidth = cs_fdp(energy_vec_bandwidth)

    cs_fdp_bandwidth = CubicSpline(energy_vec_bandwidth, fdp_vec_bandwidth)
    coeff_bandwidth = cs_fdp_bandwidth.c


    # Fit interval_0 with a polynomial, enforce endpoint continuity with interval_1
    energy_vec_0 = energy_vec[energy_vec <= interval_0[1]]
    fdp_vec_0 = fdp_vec[energy_vec <= interval_0[1]]
    if energy_vec_0[-1] != interval_0[1]:
        energy_vec_0 = np.append(energy_vec_0, interval_0[1])
        fdp_vec_0 = np.append(fdp_vec_0, cs_fdp_bandwidth(interval_0[1]))

    shift_0 = energy_vec_0[-1]
    constant_0 = fdp_vec_0[-1]

    func_fixed_pt_0 = lambda x, a, b, c, d, e: func(x, shift_0, constant_0, a, b, c, d, e)

    popt_0, pcov_0 = curve_fit(func_fixed_pt_0, energy_vec_0, fdp_vec_0)   

    # Fit interval_2 with a polynomial, enforce endpoint continuity with interval_1
    energy_vec_2 = energy_vec[energy_vec >= interval_2[0]]
    fdp_vec_2 = fdp_vec[energy_vec >= interval_2[0]]
    if energy_vec_2[0] != interval_2[0]:
        energy_vec_2 = np.append(interval_2[0], energy_vec_2)
        fdp_vec_2 = np.append(cs_fdp_bandwidth(interval_2[0]), fdp_vec_2)

    shift_2 = energy_vec_2[0]
    constant_2 = fdp_vec_2[0]

    func_fixed_pt_2 = lambda x, a, b, c, d, e: func(x, shift_2, constant_2, a, b, c, d, e)

    popt_2, pcov_2 = curve_fit(func_fixed_pt_2, energy_vec_2, fdp_vec_2)   

    # Get the entire fit on fdp

    energy_vec_full = np.concatenate((energy_vec_0, energy_vec_bandwidth, energy_vec_2), axis=0)
    fdp_vec_full = np.concatenate((func_fixed_pt_0(energy_vec_0, *popt_0), cs_fdp_bandwidth(energy_vec_bandwidth), func_fixed_pt_2(energy_vec_2, *popt_2)), axis=0)


    # Create intervals_mat, coeff_mat, powers_mat

    # interval_1 is broken up into intervals depending on step size
    interval_1_starts = np.arange(interval_1[0], interval_1[1], step_size)
    interval_1_ends = np.arange(interval_1[0]+step_size, interval_1[1] + step_size, step_size)
    interval_1_all = np.array([interval_1_starts, interval_1_ends]).T

    intervals_mat = np.concatenate([np.expand_dims(interval_0,axis=0), interval_1_all, np.expand_dims(interval_2, axis=0)],axis=0) # intervals x endpoints


    powers_mat = np.array([-2,-1,0,1,2,3])
    powers_mat = np.repeat(np.expand_dims(powers_mat, axis=0), len(intervals_mat), axis=0) # intervals x powers

    coeff_vec_0 = np.expand_dims(convert_coeff(shift_0, constant_0, *popt_0), axis=0)

    coeff_vec_1 = []
    for i in range(len(interval_1_all)):
        coeff_vec_1.append(convert_coeff(interval_1_all[i,0], coeff_bandwidth[3,i], coeff_bandwidth[0,i], coeff_bandwidth[1,i], coeff_bandwidth[2,i], 0, 0))
    coeff_vec_1 = np.stack(coeff_vec_1, axis=0)
    coeff_vec_2 = np.expand_dims(convert_coeff(shift_2, constant_2, *popt_2), axis=0)

    coeff_mat = np.concatenate([coeff_vec_0, coeff_vec_1, coeff_vec_2], axis=0)

    plot_fit(powers_mat, coeff_mat, intervals_mat, energy_vec, fdp_vec)

    # Now convert fdp to fp, account for relativistic correction

    # energy_vec_bandwidth_final = np.arange(bandedge-50, bandedge + 50, 1.)
    energy_vec_bandwidth_final = np.concatenate((np.arange(energy_vec[0]+0.5,bandedge-50,100.), np.arange(bandedge-50, bandedge + 50,1.0), np.arange(bandedge + 50, energy_vec[-1], 100.)))
    fdp_calculate_bandwidth = []
    fp_calculate_bandwidth = []

    for energy in energy_vec_bandwidth_final:
        fdp_calculate_bandwidth.append(find_fdp(energy, powers_mat, coeff_mat, intervals_mat))
        fp = fdp_fp_integrate(energy, intervals_mat, coeff_mat, powers_mat, relativistic_correction)
        fp_calculate_bandwidth.append(fp)

    fdp_calculate_bandwidth = np.array(fdp_calculate_bandwidth)
    fp_calculate_bandwidth = np.array(fp_calculate_bandwidth)
    return cs_fdp, energy_vec_bandwidth, fdp_vec_bandwidth, cs_fdp_bandwidth, energy_vec_0, \
           fdp_vec_0, func_fixed_pt_0, popt_0, energy_vec_2, fdp_vec_2, func_fixed_pt_2, popt_2, energy_vec_full, \
           fdp_vec_full, shift_0, constant_0, fp_calculate_bandwidth, energy_vec_bandwidth_final, fdp_calculate_bandwidth
