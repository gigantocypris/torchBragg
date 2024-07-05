import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
import torch
from torchBragg.kramers_kronig.cubic_spline_torch import natural_cubic_spline_coeffs_without_missing_values


def create_energy_vec(nchannels=5, mean_energy=6550, channel_width=10, library=torch):
    """
    params.spectrum.nchannels in the phil file is nchannels
    params.beam.mean_energy in the phil file is mean_energy, also this is referred to as the bandedge
    params.spectrum.channel_width in the phil file is channel_width
    """
    centerline = float(nchannels-1)/2.0
    channel_mean_eV = (library.arange(nchannels) - centerline) * channel_width + mean_energy
    return channel_mean_eV

def func(x, shift, constant, a, b, c, d, e):
    """
    function describing the fdp curve before and after the bandwidth
    shift is the end/start energy value of the curve and constant is the end/start corresponding fdp value
    shift and constant are determined by the bandwidth curve and are not directly optimizable
    a, b, c, d, e are the optimizable parameters
    """
    return a*(x-shift)**3 + b*(x-shift)**2 + c*(x-shift) + d*(x)**(-1) - d*(shift)**(-1) + e*(x)**(-2) - e*(shift)**(-2) + constant

def convert_coeff(shift, constant, a, b, c, d, e):
    """
    Inputs are in the form given by func(...)
    This function converts to the form of f*x**(-2) + g*x**(-1) + h*x**0 + i*x**1 + j*x**2 + k*x**3
    """
    f = e
    g = d
    h = -c*shift + b*shift**2 - a*shift**3 - d*(shift)**(-1) - e*(shift)**(-2) + constant
    i = c - b*2*shift + a*3*shift**2
    j = b - a*3*shift
    k = a


    return torch.tensor([f,g,h,i,j,k])


def func_bandwidth(x, interval_inds, energy_vec_bandwidth, fdp_vec_base):
    """
    Cubic splines with natural boundary conditions (i.e. second derivatives at the very beginning and very end are 0)
    energy_vec_bandwidth is offset from the places where we want to interpolate fdp and calculate fp
    interval is the index of energy_vec_bandwidth that is lower than x, with energy_vec_bandwidth[index+1] being higher than x
    """
    constant, c, b, a = natural_cubic_spline_coeffs_without_missing_values(energy_vec_bandwidth, fdp_vec_base)

    shift = energy_vec_bandwidth[interval_inds]
    d = torch.zeros_like(shift)
    e = torch.zeros_like(shift)

    coeff = torch.stack((shift, constant, a, b, c, d, e))
    fdp_values = a[interval_inds]*(x-shift)**3 + b[interval_inds]*(x-shift)**2 + c[interval_inds]*(x-shift) + constant
    return fdp_values, coeff

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
        integral += coeff_i*((-energy)**(n+1))*torch.log(torch.abs((energy_end + energy)/(energy_start+energy)))
        if n <= -2:
            integral += -coeff_i*((-energy)**(n+1))*torch.log(torch.abs(energy_end/energy_start))
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
            integral += coeff_i*(energy**(n+1))*torch.log(torch.abs((energy_end - energy)/(energy_start - energy))) ## Problem term here
        except FloatingPointError as rw:
            print("FloatingPointError:", rw)
            print("Check that energy is not at the endpoints of any interval")
        if n <= -2:
            integral += -coeff_i*(energy**(n+1))*torch.log(torch.abs(energy_end/energy_start))
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


def get_free_params(energy_vec_reference, fdp_vec_reference, energy_vec_bandwidth):
    """
    This function is used to get the initial conditions for the free parameters from published curves.

    energy_vec_reference and fdp_vec_reference are the published curves
    energy_vec_bandwidth corresponds to the interval endpoints in the bandwidth, these must be offset
    from the actual points we want to determine fdp and fp at, otherwise we have a singularity in
    the Kramers-Kronig formulation
    """

    # Fit the entire curve with a cubic spline
    cs_fdp = CubicSpline(energy_vec_reference, fdp_vec_reference)

    params_bandwidth = cs_fdp(energy_vec_bandwidth)

    # Params for the beginning of the curve
    energy_vec_0 = energy_vec_reference[energy_vec_reference <= energy_vec_bandwidth[0]]
    breakpoint()
    fdp_vec_0 = fdp_vec_reference[energy_vec_reference <= energy_vec_bandwidth[0]]
    shift_0 = energy_vec_bandwidth[0]
    constant_0 = params_bandwidth[0]

    func_fixed_pt_0 = lambda x, a, b, c, d, e: func(x, shift_0, constant_0, a, b, c, d, e)
    popt_0, pcov_0 = curve_fit(func_fixed_pt_0, energy_vec_0, fdp_vec_0)   

    # Params for the end of the curve
    energy_vec_1 = energy_vec_reference[energy_vec_reference >= energy_vec_bandwidth[-1]]
    fdp_vec_1 = fdp_vec_reference[energy_vec_reference >= energy_vec_bandwidth[-1]]
    shift_1 = energy_vec_bandwidth[-1]
    constant_1 = params_bandwidth[-1]

    func_fixed_pt_1 = lambda x, a, b, c, d, e: func(x, shift_1, constant_1, a, b, c, d, e)
    popt_1, pcov_1 = curve_fit(func_fixed_pt_1, energy_vec_1, fdp_vec_1)   


    # Get the entire fit on fdp
    energy_vec_full = np.concatenate((energy_vec_0, energy_vec_bandwidth, energy_vec_1), axis=0)
    fdp_vec_full = np.concatenate((func_fixed_pt_0(energy_vec_0, *popt_0), params_bandwidth, func_fixed_pt_1(energy_vec_1, *popt_1)), axis=0)

    # params = np.concatenate((popt_0, params_bandwidth, popt_1))
    return popt_0, params_bandwidth, popt_1, shift_0, constant_0, shift_1, constant_1, energy_vec_full, fdp_vec_full

# def find_fdp(energy, powers_mat, coeff_mat, intervals_mat):
#     """
#     Evaluates the values for fdp for the full curve
#     For checking the fdp fit for the full curve, useful for plotting and verifying initial conditions
#     """
#     # find what interval energy is in
#     interval_ind = intervals_mat - energy # the interval that energy is in has a change from negative to positive
#     interval_ind[interval_ind<=0] = 0 # equal sign should never be triggered!
#     interval_ind[interval_ind>0] = 1
#     interval_ind = interval_ind.astype(bool)
#     interval_ind = np.sum(interval_ind, axis=1)*(1-np.prod(interval_ind, axis=1)) # XOR
#     interval_ind = np.where(interval_ind)[0][0]

#     power = powers_mat[interval_ind]
#     coeff = coeff_mat[interval_ind]
#     interval = intervals_mat[interval_ind]
#     fdp_fit = func_converted_coeff(energy, power, coeff)
#     return fdp_fit

def get_physical_params_fdp(energy_vec, energy_vec_bandwidth, params):
    """
    This function gets the fdp physical parameters from params

    energy_vec are the energies we want to evaluate at, energy_vec_bandwidth are offset and where the fdp_vec_base values are
    this offset is necessary to evaluate the integral to compute fp_vec at energy_vec energies

    params is in the following form:
    func_start_end       -- func_bandwith  -- func_start_end
    [a0, b0, c0, d0, e0] -- [fdp_vec_base] -- [a1, b1, c1, d1, e1]
    """
    a0, b0, c0, d0, e0 = params[0:5]
    fdp_vec_base = params[5:-5]
    a1, b1, c1, d1, e1 = params[-5:]

    shift0 = energy_vec_bandwidth[0]
    constant0 = fdp_vec_base[0]

    shift1 = energy_vec_bandwidth[-1]
    constant1 = fdp_vec_base[-1]

    interval_inds = torch.arange(len(energy_vec), dtype=torch.int32)

    fdp0 = func(energy_vec[energy_vec < shift0], shift0, constant0, a0, b0, c0, d0, e0)
    fdp_bandwidth, coeff_vec_bandwidth = func_bandwidth(energy_vec, interval_inds, energy_vec_bandwidth, fdp_vec_base)
    fdp1 = func(energy_vec[energy_vec > shift1], shift1, constant1, a1, b1, c1, d1, e1)
    breakpoint()
    fdp_full = torch.concat((fdp0, fdp_bandwidth, fdp1))

    return fdp_full, coeff_vec_bandwidth

def get_physical_params_fp(energy_vec, energy_vec_bandwidth, params, coeff_vec_bandwidth, relativistic_correction):
    """
    This function gets the fp physical parameters from params

    energy_vec are the energies we want to evaluate at, energy_vec_bandwidth are offset and where the fdp_vec_base values are
    this offset is necessary to evaluate the integral to compute fp_vec at energy_vec energies

    params is in the following form:
    func_start_end       -- func_bandwith  -- func_start_end
    [a0, b0, c0, d0, e0] -- [fdp_vec_base] -- [a1, b1, c1, d1, e1]
    """
    a0, b0, c0, d0, e0 = params[0:5]
    fdp_vec_base = params[5:-5]
    a1, b1, c1, d1, e1 = params[-5:]

    shift0 = energy_vec_bandwidth[0]
    constant0 = fdp_vec_base[0]

    shift1 = energy_vec_bandwidth[-1]
    constant1 = fdp_vec_base[-1]

    # Create intervals_mat, coeff_mat, powers_mat
    interval_0 = torch.tensor([0, energy_vec_bandwidth[0]])
    interval_bandwidth = torch.stack((energy_vec_bandwidth[:-1], energy_vec_bandwidth[1:]),axis=0)
    interval_1 = torch.tensor([energy_vec_bandwidth[-1], 10000])

    # # interval_bandwidth is broken up into intervals depending on step size
    # interval_bandwidth_starts = energy_vec_bandwidth[:-1]
    # interval_bandwidth_ends = energy_vec_bandwidth[1:]
    # interval_bandwidth_all = np.array([interval_bandwidth_starts, interval_bandwidth_ends]).T

    # intervals_mat = np.concatenate([np.expand_dims(interval_0,axis=0), interval_bandwidth_all, np.expand_dims(interval_1, axis=0)],axis=0) # intervals x endpoints

    intervals_mat = torch.concat((interval_0, interval_bandwidth, interval_1)) # intervals x endpoints

    powers_mat = torch.tensor([-2,-1,0,1,2,3])
    powers_mat = torch.repeat(torch.expand_dims(powers_mat, axis=0), len(intervals_mat), axis=0) # intervals x powers

    coeff_vec_0 = torch.expand_dims(convert_coeff(shift0, constant0, *params[0:5]), axis=0)

    coeff_vec_bandwidth = []
    for i in range(len(interval_bandwidth)):
        coeff_vec_bandwidth.append(convert_coeff(interval_bandwidth[i,0], *coeff_vec_bandwidth[:,i]))
    coeff_vec_bandwidth = torch.stack(coeff_vec_bandwidth, axis=0)

    breakpoint()
    # coeff_vec_1 = np.expand_dims(convert_coeff(shift1, constant1, *params[-5:]), axis=0)
    coeff_vec_1 = convert_coeff(shift1, constant1, *params[-5:])[None]

    coeff_mat = torch.concat([coeff_vec_0, coeff_vec_bandwidth, coeff_vec_1], axis=0)


    # Now convert fdp to fp, account for relativistic correction

    fp_full = []

    for energy in energy_vec:
        fp = fdp_fp_integrate(energy, intervals_mat, coeff_mat, powers_mat, relativistic_correction)
        fp_full.append(fp)
    
    return fp_full



def convert_fdp_to_fp(energy_vec_reference, energy_vec_bandwidth, fdp_vec_reference, relativistic_correction):
    # energy_vec_bandwidth cannot have any points in energy_vec_reference

    popt_0, params_bandwidth, popt_1, shift_0, constant_0, shift_1, constant_1, energy_vec_full, fdp_vec_full = get_free_params(energy_vec_reference, fdp_vec_reference, energy_vec_bandwidth)
    params = torch.tensor(np.concatenate((popt_0, params_bandwidth, popt_1)))

    energy_vec_reference = torch.tensor(energy_vec_reference)
    energy_vec_bandwidth = torch.tensor(energy_vec_bandwidth)

    fdp_full, coeff_vec_bandwidth = get_physical_params_fdp(energy_vec_reference, energy_vec_bandwidth, params)
    fp_full = get_physical_params_fp(energy_vec_reference, energy_vec_bandwidth, params, coeff_vec_bandwidth, relativistic_correction)

    return fdp_full, fp_full
