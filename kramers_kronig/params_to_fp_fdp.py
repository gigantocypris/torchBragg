"""
PyTorch function that takes in optimizable params
(as well as input parameters mean_energy, nchannels, channel_width) and outputs fp and fdp.

Usage: libtbx.python $MODULES/torchBragg/SPREAD_integration/params_to_fp_fdp.py
"""
import torch
from cubic_spline_torch import natural_cubic_spline_coeffs_without_missing_values
from convert_fdp_helper import fdp_fp_integrate

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
    
def func_bandwidth(x, interval_inds, energy_vec_base, fdp_vec_base):
    """
    Cubic splines with natural boundary conditions (i.e. second derivatives at the very beginning and very end are 0)
    energy_vec_base is offset from the places where we want to interpolate fdp and calculate fp
    interval is the index of energy_vec_base that is lower than x, with energy_vec_base[index+1] being higher than x
    """
    constant, c, b, a = natural_cubic_spline_coeffs_without_missing_values(energy_vec_base, fdp_vec_base)

    shift = energy_vec_base[interval_inds]
    d = torch.zeros_like(shift)
    e = torch.zeros_like(shift)

    coeff = torch.stack((shift, constant, a, b, c, d, e))
    fdp_values = a[interval_inds]*(x-shift)**3 + b[interval_inds]*(x-shift)**2 + c[interval_inds]*(x-shift) + constant
    return fdp_values, coeff


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

    interval_inds = torch.arange(len(energy_vec), dtype=torch.int32)

    # fdp0 = func_start_end(energy_vec, shift0, constant0, a0, b0, c0, d0, e0)
    fdp_bandwidth, coeff_bandwidth = func_bandwidth(energy_vec, interval_inds, energy_vec_base, fdp_vec_base)
    # fdp1 = func_start_end(energy_vec, shift1, constant1, a1, b1, c1, d1, e1)

    return fdp_bandwidth, coeff_bandwidth

def calculate_fp(energy_vec, energy_vec_base, params, coeff_bandwidth):
    # a0, b0, c0, d0, e0 = params[0:5]
    fdp_vec_base = params[5:-5]
    # a1, b1, c1, d1, e1 = params[-5:]

    shift0 = energy_vec_base[0]
    constant0 = fdp_vec_base[0]

    shift1 = energy_vec_base[-1]
    constant1 = fdp_vec_base[-1]

    # intervals = torch.arange(len(energy_vec), dtype=torch.int32)

    # fdp0 = func_start_end(energy_vec, shift0, constant0, a0, b0, c0, d0, e0)
    # fdp_bandwidth = func_bandwidth(energy_vec, intervals, energy_vec_base, fdp_vec_base)
    # fdp1 = func_start_end(energy_vec, shift1, constant1, a1, b1, c1, d1, e1)



    # Create intervals_mat, coeff_mat, powers_mat
    interval_0 = torch.tensor([0, energy_vec_base[0]])
    interval_bandwidth = torch.stack((energy_vec_base[:-1], energy_vec_base[1:]),axis=0)
    interval_1 = torch.tensor([energy_vec_base[-1], 10000])



    intervals_mat = torch.concat((interval_0, interval_bandwidth, interval_1)) # intervals x endpoints


    powers_mat = torch.tensor([-2,-1,0,1,2,3])
    powers_mat = torch.repeat(torch.expand_dims(powers_mat, axis=0), len(intervals_mat), axis=0) # intervals x powers

    coeff_vec_0 = torch.expand_dims(convert_coeff(shift_0, constant_0, *params[0:5]), axis=0)

    coeff_vec_bandwidth = []
    for i in range(len(interval_bandwidth)):
        coeff_vec_bandwidth.append(convert_coeff(interval_bandwidth[i,0], *coeff_bandwidth[:,i]))
    coeff_vec_bandwidth = torch.stack(coeff_vec_bandwidth, axis=0)

    coeff_vec_1 = np.expand_dims(convert_coeff(shift_1, constant_1, *params[-5:]), axis=0)

    coeff_mat = np.concatenate([coeff_vec_0, coeff_vec_bandwidth, coeff_vec_1], axis=0)


    # Now convert fdp to fp, account for relativistic correction

    # energy_vec_bandwidth_final = np.arange(bandedge-50, bandedge + 50, 1.)
    energy_vec_bandwidth_final = np.concatenate((np.arange(energy_vec[0]+0.5,bandedge-50,100.), np.arange(bandedge-50, bandedge + 50,1.0), np.arange(bandedge + 50, energy_vec[-1], 100.)))
    fp_calculate_bandwidth = []

    for energy in energy_vec:
        fp = fdp_fp_integrate(energy, intervals_mat, coeff_mat, powers_mat, relativistic_correction)
        fp_calculate_bandwidth.append(fp)
    
    return fp_calculate_bandwidth

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