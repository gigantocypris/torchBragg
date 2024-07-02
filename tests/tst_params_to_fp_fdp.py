"""
Usage:
libtbx.python $MODULES/torchBragg/tests/tst_params_to_fp_fdp.py
"""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import torch
from torchBragg.kramers_kronig.create_fp_fdp_dat_file import full_path, read_dat_file
from torchBragg.kramers_kronig.params_to_fp_fdp import create_energy_vec, func_start_end, evaluate_fdp, calculate_fp

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
fdp_vec, coeff_bandwidth = evaluate_fdp(energy_vec, energy_vec_base_torch, params_torch)

plt.figure()
plt.plot(energy_vec_reference, fdp_vec_reference)
plt.plot(energy_vec, fdp_vec, '.')
plt.xlim([energy_vec_base[0], energy_vec_base[-1]])
plt.savefig("test_params.png")

calculate_fp(energy_vec, energy_vec_base, params, coeff_bandwidth)