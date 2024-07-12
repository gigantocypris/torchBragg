import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
import torch
from torchBragg.kramers_kronig.cubic_spline_torch import natural_cubic_spline_coeffs_without_missing_values

def check_clashes(energy_vec_0, energy_vec_1):
    """
    checks if there are any common values in energy_vec_0 and energy_vec_1
    """
    
    return len(set(energy_vec_0).intersection(set(energy_vec_1)))

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
    function can describe the fdp curve within the bandwidth through the use of get_coeff_bandwidth
    shift is the end/start energy value of the curve and constant is the end/start corresponding fdp value
    shift and constant are determined by the bandwidth curve and are not directly optimizable
    a, b, c, d, e are the optimizable parameters
    """
    return a*(x-shift)**3 + b*(x-shift)**2 + c*(x-shift) + d*(x)**(-1) - d*(shift)**(-1) + e*(x)**(-2) - e*(shift)**(-2) + constant

def func_reformatted(x, power, coeff):
    return torch.sum(coeff*(x**power), axis=1)

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
    return torch.stack([f,g,h,i,j,k],axis=1)

def get_coeff_bandwidth(energy_vec_bandwidth, fdp_vec_base):
    """
    Get coefficients for the bandwidth intervals in the form given by func(...)
    """
    constant, c, b, a = natural_cubic_spline_coeffs_without_missing_values(energy_vec_bandwidth, fdp_vec_base)
    d = torch.zeros_like(constant)
    e = torch.zeros_like(constant)
    coeff = torch.stack((energy_vec_bandwidth[:-1], constant, a, b, c, d, e))
    return coeff

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

def find_interval_inds(energy_vec_bandwidth, energy_vec):
    """
    energy_vec_bandwidth are the points the free parameters are calculated at
    energy_vec are the physical parameters, points where we want to convert fdp to fp
    we want to find what interval index each value of energy_vec is in, for energy_vec[i],
    the interval_ind is where energy_vec_bandwidth[interval_ind] < energy_vec[i] < energy_vec_bandwidth[interval_ind+1]
    Note that there are no equality signs as there can be no clashes in the values of energy_vec_bandwidth and energy_vec
    """
    interval_inds = energy_vec_bandwidth[:,None]- energy_vec[None,:] # the interval that energy_vec_bandwidth is in has a change from negative to positive
    interval_inds[interval_inds<=0] = 0 # equal sign should never be triggered!
    interval_inds[interval_inds>0] = 1
    interval_inds = interval_inds.type(dtype=torch.bool)
    interval_inds = len(energy_vec_bandwidth) - 1 - torch.sum(interval_inds, axis=0)
    return interval_inds

def get_physical_params_fdp(energy_vec, energy_vec_bandwidth, free_params, coeff_vec_bandwidth):
    """
    This function gets the fdp physical parameters from the free params

    energy_vec are the energies we want to evaluate at, energy_vec_bandwidth are offset and where the fdp_vec_base values are
    this offset is necessary to evaluate the integral to compute fp_vec at energy_vec energies

    params is in the following form:
    func_start_end       -- func_bandwith  -- func_start_end
    [a0, b0, c0, d0, e0] -- [fdp_vec_base] -- [a1, b1, c1, d1, e1]
    """
    a0, b0, c0, d0, e0 = free_params[0:5]
    fdp_vec_base = free_params[5:-5]
    a1, b1, c1, d1, e1 = free_params[-5:]

    shift0 = energy_vec_bandwidth[0]
    constant0 = fdp_vec_base[0]

    shift1 = energy_vec_bandwidth[-1]
    constant1 = fdp_vec_base[-1]

    interval_inds = find_interval_inds(energy_vec_bandwidth, energy_vec[(energy_vec >= shift0) & (energy_vec <= shift1)])
    fdp0 = func(energy_vec[energy_vec < shift0], shift0, constant0, a0, b0, c0, d0, e0)
    fdp_bandwidth = func(energy_vec[(energy_vec >= shift0) & (energy_vec <= shift1)], *coeff_vec_bandwidth[:,interval_inds])
    fdp1 = func(energy_vec[energy_vec > shift1], shift1, constant1, a1, b1, c1, d1, e1)
    fdp_full = torch.concat((fdp0, fdp_bandwidth, fdp1))
    
    return fdp_full

def reformat_fdp(energy_vec_bandwidth, free_params, coeff_vec_bandwidth):
    """
    This function reformats the parameters describing the fdp curve in order to later calculate the fp values.

    energy_vec are the energies we want to evaluate at, energy_vec_bandwidth are offset and where the fdp_vec_base values are
    this offset is necessary to evaluate the integral to compute fp_vec at energy_vec energies

    params is in the following form:
    func_start_end       -- func_bandwith  -- func_start_end
    [a0, b0, c0, d0, e0] -- [fdp_vec_base] -- [a1, b1, c1, d1, e1]
    """

    a0, b0, c0, d0, e0 = free_params[0:5]
    fdp_vec_base = free_params[5:-5]
    a1, b1, c1, d1, e1 = free_params[-5:]

    shift0 = energy_vec_bandwidth[0]
    constant0 = fdp_vec_base[0]

    shift1 = energy_vec_bandwidth[-1]
    constant1 = fdp_vec_base[-1]

    # Create intervals_mat, coeff_mat, powers_mat
    interval_0 = torch.tensor([800, energy_vec_bandwidth[0]])[None]
    interval_bandwidth = torch.stack((energy_vec_bandwidth[:-1], energy_vec_bandwidth[1:]), axis=1)
    interval_1 = torch.tensor([energy_vec_bandwidth[-1], 30000])[None]

    intervals_mat = torch.concat((interval_0, interval_bandwidth, interval_1), axis=0) # intervals x endpoints

    powers_mat = torch.tensor([-2,-1,0,1,2,3])
    powers_mat = powers_mat.repeat(len(intervals_mat),1) # intervals x powers

    free_params_0 = free_params[0:5][:,None]
    coeff_vec_0 = convert_coeff(shift0[None], constant0[None], *free_params_0)
    coeff_vec_bandwidth_converted = convert_coeff(*coeff_vec_bandwidth)

    free_params_1 = free_params[-5:][:,None]
    coeff_vec_1 = convert_coeff(shift1[None], constant1, *free_params_1)

    coeff_mat = torch.concat([coeff_vec_0, coeff_vec_bandwidth_converted, coeff_vec_1], axis=0) # intervals x coefficients
    return intervals_mat, coeff_mat, powers_mat


def get_reformatted_fdp(energy, powers_mat, coeff_mat, intervals_mat):
    """
    Evaluates the values for fdp for the full curve
    For checking the fdp fit for the full curve, useful for plotting and verifying initial conditions
    """
    # find what interval energy is in
    intervals = torch.concat((intervals_mat[:,0],intervals_mat[:,1][-1][None])) # must be monotonically increasing
    interval_ind = find_interval_inds(intervals, energy)

    power = powers_mat[interval_ind]
    coeff = coeff_mat[interval_ind]
    fdp = func_reformatted(energy[:,None], power, coeff)
    return fdp

def get_physical_params_fp(energy_vec, energy_vec_bandwidth, free_params, coeff_vec_bandwidth, relativistic_correction):
    """
    This function gets the fp physical parameters from free_params

    energy_vec are the energies we want to evaluate at, energy_vec_bandwidth are offset and where the fdp_vec_base values are
    this offset is necessary to evaluate the integral to compute fp_vec at energy_vec energies

    params is in the following form:
    func_start_end       -- func_bandwith  -- func_start_end
    [a0, b0, c0, d0, e0] -- [fdp_vec_base] -- [a1, b1, c1, d1, e1]
    """
    intervals_mat, coeff_mat, powers_mat = reformat_fdp(energy_vec_bandwidth, free_params, coeff_vec_bandwidth)

    # Now convert fdp to fp, account for relativistic correction

    fp_full = []
    for energy in energy_vec:
        print(energy)
        fp = fdp_fp_integrate(energy, intervals_mat, coeff_mat, powers_mat, relativistic_correction)
        fp_full.append(fp)
    
    return fp_full

def convert_fdp_to_fp(energy_vec_reference, energy_vec_bandwidth, energy_vec_final, fdp_vec_reference, relativistic_correction):
    """ 
    energy_vec_reference is the x-values for fdp_vec_reference
    energy_vec_bandwidth denotes the x-values of the points we in the bandwidth that define the cubic spline fit there
    energy_vec_final are the x-values for the physical parameters fdp and fp 
    
    energy_vec_bandwidth cannot have any points in energy_vec_reference
    """
    popt_0, params_bandwidth, popt_1, shift_0, constant_0, shift_1, constant_1, energy_vec_full, fdp_vec_full = \
        get_free_params(energy_vec_reference, fdp_vec_reference, energy_vec_bandwidth)
    
    
    free_params = torch.tensor(np.concatenate((popt_0, params_bandwidth, popt_1)))

    energy_vec_bandwidth = torch.tensor(energy_vec_bandwidth)
    coeff_vec_bandwidth = get_coeff_bandwidth(energy_vec_bandwidth, torch.tensor(params_bandwidth))

    fdp_final = get_physical_params_fdp(energy_vec_final, energy_vec_bandwidth, free_params, coeff_vec_bandwidth)    
    fp_final = get_physical_params_fp(energy_vec_final, energy_vec_bandwidth, free_params, coeff_vec_bandwidth, relativistic_correction)

    intervals_mat, coeff_mat, powers_mat = reformat_fdp(energy_vec_bandwidth, free_params, coeff_vec_bandwidth)
    fdp_check = get_reformatted_fdp(energy_vec_final, powers_mat, coeff_mat, intervals_mat)
    print('Numerical Check:', torch.allclose(fdp_final, fdp_check, rtol=1e-05, atol=1e-08, equal_nan=False))

    return fdp_final, fp_final
