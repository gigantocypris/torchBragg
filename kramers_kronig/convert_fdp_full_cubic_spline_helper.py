import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
import torch
from torchBragg.kramers_kronig.cubic_spline_torch import natural_cubic_spline_coeffs_without_missing_values
from torchBragg.kramers_kronig.convert_fdp_helper import find_interval_inds

def create_energy_vec_free(energies, start_energy_free=1000.0, end_energy_free=24900.0, subsample=1):
    # creates energy_vec_free, points where the FREE PARAMETERS are located
    # energies are where PHYSICAL PARAMETERS are located
    # energy_vec_free must be different at all points to energies
    # energies = np.array(energies)
    # energies = energies[np.array([0,10,15,20, 21, 22, 23, 24, 25, 26, 27, 28, 29,39,44, 49])]
    d_energy = np.array(energies[1:]) - np.array(energies[:-1])
    energy_vec_bandwidth = np.zeros(len(energies)+1)
    energy_vec_bandwidth[1:-1] = energies[:-1] + d_energy/2
    energy_vec_bandwidth[-1] = energies[-1] + d_energy[-1]/2
    energy_vec_bandwidth[0] = energies[0] - d_energy[0]/2
    energy_vec_bandwidth_subsample = energy_vec_bandwidth[1:-1][::subsample]
    energy_start = np.arange(start_energy_free, energy_vec_bandwidth[0],100)
    energy_end = np.arange(energy_vec_bandwidth[-1], end_energy_free, 100)
    energy_vec_free = np.concatenate((energy_start, energy_vec_bandwidth_subsample, energy_end))
    mask = np.concatenate((np.zeros(len(energy_start)+2), np.hanning(len(energy_vec_bandwidth_subsample)-4), np.zeros(len(energy_end)+2)))
    return energy_vec_free, mask


def get_free_params(energy_vec_reference, fdp_vec_reference, energy_vec_free):
    """
    This function is used to get the initial conditions for the free parameters from published curves.

    energy_vec_reference and fdp_vec_reference are the published curves
    energy_vec_free corresponds to the interval endpoints, these must be offset
    from the actual points we want to determine fdp and fp at, otherwise we have a singularity in
    the Kramers-Kronig formulation
    """
    # Fit the entire curve with a cubic spline
    cs_fdp = CubicSpline(energy_vec_reference, fdp_vec_reference)

    fdp_vec_free = cs_fdp(energy_vec_free)
    return fdp_vec_free

def get_coeff_cubic_spline(energy_vec_free, fdp_vec_free):
    """
    Get coefficients for the cubic spline
    """
    constant, c, b, a = natural_cubic_spline_coeffs_without_missing_values(energy_vec_free, fdp_vec_free)
    params_free_cubic_spline = torch.stack((energy_vec_free[:-1], constant, a, b, c))
    return params_free_cubic_spline


def reformat_fdp(energy_vec_free, params_free_cubic_spline, device='cpu'):
    """
    This function reformats the parameters describing the fdp curve in order to later calculate the fp values,
    creating intervals_mat, coeff_mat, powers_mat.
    """

    intervals_mat = torch.stack((energy_vec_free[:-1], energy_vec_free[1:]), axis=1)
    intervals_mat = intervals_mat.to(device) # intervals x endpoints

    powers_mat = torch.tensor([0,1,2,3], device=device)
    powers_mat = powers_mat.repeat(len(intervals_mat),1) # intervals x powers

    coeff_mat = convert_coeff(*params_free_cubic_spline) # intervals x coefficients

    return intervals_mat, coeff_mat, powers_mat

def convert_coeff(shift, constant, a, b, c):
    """
    Inputs are in the form given by func(...)
    This function converts to the form of h*x**0 + i*x**1 + j*x**2 + k*x**3
    """
    h = -c*shift + b*shift**2 - a*shift**3 + constant
    i = c - b*2*shift + a*3*shift**2
    j = b - a*3*shift
    k = a
    return torch.stack([h,i,j,k],axis=1)

def get_physical_params_fdp(energy_vec_physical, energy_vec_free, params_free_cubic_spline):
    """
    This function gets the fdp physical parameters from the free params (params_free_cubic_spline)

    energy_vec_physical are the energies we want to evaluate at, energy_vec_free are offset and where the params_free_cubic_spline values are
    this offset is necessary to evaluate the integral to compute fp at energy_vec_physical energies downstream

    """
    interval_inds = find_interval_inds(energy_vec_free, energy_vec_physical) # find the correct interval for each energy_vec_physical value

    fdp = func(energy_vec_physical, *params_free_cubic_spline[:,interval_inds])
    
    return fdp

def func(x, shift, constant, a, b, c):
    """
    function to describe the fdp curve 
    shift is the end/start energy value of the curve and constant is the end/start corresponding fdp value
    """
    return a*(x-shift)**3 + b*(x-shift)**2 + c*(x-shift) + constant
