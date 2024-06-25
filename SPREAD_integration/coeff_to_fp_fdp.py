"""
Write PyTorch function that takes in coefficients (as well as input parameters mean_energy, nchannels, channel_width) and outputs fp and fdp

Usage: libtbx.python $MODULES/torchBragg/SPREAD_integration/coeff_to_fp_fdp.py
"""
import torch
from helper import natural_cubic_spline_coeffs_without_missing_values

"""
params.spectrum.nchannels
params.spectrum.channel_width
params.beam.mean_energy
"""
def create_energy_vec(nchannels=5, mean_energy=6550, channel_width=10, prefix=torch):
    centerline = float(nchannels-1)/2.0
    channel_mean_eV = (prefix.arange(nchannels) - centerline) * channel_width + mean_energy
    return channel_mean_eV

def func_start_end(x, shift, constant, a, b, c, d, e):
    """
    function describing the fdp curve before and after the bandwidth
    shift is the end/start energy value of the curve and constant is the end/start corresponding fdp value
    shift and constant are determined by the bandwidth curve and are not directly optimizable
    a, b, c, d, e are the optimizable parameters
    """
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
    
def func_bandwidth(x, intervals, energy_vec_base, fdp_vec_base):
    """
    Cubic splines with natural boundary conditions (i.e. second derivatives at the very beginning and very end are 0)
    energy_vec_base is offset from the places where we want to interpolate fdp and calculate fp
    interval is the index of energy_vec_base that is lower than x, with energy_vec_base[index+1] being higher than x
    """
    constant, c, b, a = natural_cubic_spline_coeffs_without_missing_values(energy_vec_base, fdp_vec_base)

    shift = energy_vec_base[intervals]
    # d = 0
    # e = 0
    return a[intervals]*(x-shift)**3 + b[intervals]*(x-shift)**2 + c[intervals]*(x-shift) + constant


def evaluate_fdp(energy_vec, energy_vec_base, params):
    """
    energy_vec are the energies we want to evaluate at, energy_vec_base are offset and where the fdp_vec_base values
    this offset is necessary to evaluate the integral to compute fp_vec at energy_vec energies

    params is in the following form:
    func_start_end       -- func_bandwith  -- func_start_end
    [a0, b0, c0, d0, e0] -- [fdp_vec_base] -- [a1, b1, c1, d1, e1]
    """
    # a0, b0, c0, d0, e0 = params[0:5]
    fdp_vec_base = params[5:-5]
    # a1, b1, c1, d1, e1 = params[-5:]

    # shift0 = energy_vec_base[0]
    # constant0 = fdp_vec_base[0]

    # shift1 = energy_vec_base[-1]
    # constant1 = fdp_vec_base[-1]

    intervals = torch.arange(len(energy_vec), dtype=torch.int32)

    # fdp0 = func_start_end(energy_vec, shift0, constant0, a0, b0, c0, d0, e0)
    fdp_bandwidth = func_bandwidth(energy_vec, intervals, energy_vec_base, fdp_vec_base)
    # fdp1 = func_start_end(energy_vec, shift1, constant1, a1, b1, c1, d1, e1)

    return fdp_bandwidth

def calculate_fp(energy_vec, params):
    pass

if __name__ == '__main__':
    from helper import full_path, read_dat_file
    import numpy as np
    from scipy.interpolate import CubicSpline
    from scipy.optimize import curve_fit
    import matplotlib.pyplot as plt
    
    nchannels=5; mean_energy=6550; channel_width=10
    energy_vec = create_energy_vec(nchannels, mean_energy, channel_width)
    energy_vec_base = create_energy_vec(nchannels+1, mean_energy, channel_width, prefix=np)

    # Path to Sherrell data: $MODULES/ls49_big_data/data_sherrell
    prefix = "Mn2O3_spliced" # Fe, Mn, MnO2_spliced, Mn2O3_spliced
    Mn_model=full_path("data_sherrell/" + prefix + ".dat")
    relativistic_correction = 0 # 0.042 for Mn and 0.048 for Fe
    bandedge = 6550 # 6550 eV is the bandedge of Mn and 7112 is the bandedge of Fe


    energy_vec_reference, fp_vec_reference, fdp_vec_reference = read_dat_file(Mn_model)
    energy_vec_reference = np.array(energy_vec_reference).astype(np.float64)
    fp_vec_reference = np.array(fp_vec_reference).astype(np.float64)
    fdp_vec_reference = np.array(fdp_vec_reference).astype(np.float64)

    # Fit the entire curve with a cubic spline
    cs_fdp = CubicSpline(energy_vec_reference, fdp_vec_reference)

    # Params for the bandwith
    fdp_vec_base = cs_fdp(energy_vec_base)

    # Params for the beginning of the curve
    energy_vec_0 = energy_vec_reference[energy_vec_reference <= energy_vec_base[0]]
    fdp_vec_0 = fdp_vec_reference[energy_vec_reference <= energy_vec_base[0]]
    shift_0 = energy_vec_base[0]
    constant_0 = fdp_vec_base[0]

    func_fixed_pt_0 = lambda x, a, b, c, d, e: func_start_end(x, shift_0, constant_0, a, b, c, d, e)
    popt_0, pcov_0 = curve_fit(func_fixed_pt_0, energy_vec_0, fdp_vec_0)   

    # Params for the end of the curve
    energy_vec_1 = energy_vec_reference[energy_vec_reference >= energy_vec_base[-1]]
    fdp_vec_1 = fdp_vec_reference[energy_vec_reference >= energy_vec_base[-1]]
    shift_1 = energy_vec_base[-1]
    constant_1 = fdp_vec_base[-1]

    func_fixed_pt_1 = lambda x, a, b, c, d, e: func_start_end(x, shift_1, constant_1, a, b, c, d, e)
    popt_1, pcov_1 = curve_fit(func_fixed_pt_1, energy_vec_1, fdp_vec_1)   

    params = np.concatenate((popt_0, fdp_vec_base, popt_1))

    energy_vec_base_torch = create_energy_vec(nchannels+1, mean_energy, channel_width, prefix=torch)
    params_torch = torch.tensor(params)
    fdp_vec = evaluate_fdp(energy_vec, energy_vec_base_torch, params_torch)

    plt.figure()
    plt.plot(energy_vec_reference, fdp_vec_reference)
    plt.plot(energy_vec, fdp_vec, '.')
    plt.xlim([energy_vec_base[0], energy_vec_base[-1]])
    plt.savefig("test_params.png")

    breakpoint()