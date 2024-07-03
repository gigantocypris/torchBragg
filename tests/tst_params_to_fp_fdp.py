"""
Usage:
libtbx.python $MODULES/torchBragg/tests/tst_params_to_fp_fdp.py --prefix Mn2O3_spliced
"""
import argparse
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import torch
from torchBragg.kramers_kronig.create_fp_fdp_dat_file import full_path, read_dat_file
from torchBragg.kramers_kronig.convert_fdp_helper import create_energy_vec, get_free_params, evaluate_fdp, calculate_fp
# Create initial free parameters

parser = argparse.ArgumentParser()
parser.add_argument('--prefix', type=str, default='Mn',
                    help='filename prefix for the fp and fdp curves to load',
                    choices=['Fe', 'Mn', 'MnO2_spliced', 'Mn2O3_spliced'])
args = parser.parse_args()

# Path to Sherrell data: $MODULES/ls49_big_data/data_sherrell
prefix = args.prefix
Mn_model=full_path("data_sherrell/" + prefix + ".dat")
relativistic_correction = 0 # 0.042 for Mn and 0.048 for Fe
bandedge = 6550 # 6550 eV is the bandedge of Mn and 7112 is the bandedge of Fe


energy_vec, fp_vec, fdp_vec = read_dat_file(Mn_model)
energy_vec = np.array(energy_vec).astype(np.float64)
fp_vec = np.array(fp_vec).astype(np.float64)
fdp_vec = np.array(fdp_vec).astype(np.float64)

cs_fdp, energy_vec_bandwidth,\
fdp_vec_bandwidth, cs_fdp_bandwidth, energy_vec_0, fdp_vec_0,\
func_fixed_pt_0, popt_0, energy_vec_2, fdp_vec_2, func_fixed_pt_2,\
popt_2, energy_vec_full, fdp_vec_full, shift_0, constant_0, \
fp_calculate_bandwidth, energy_vec_bandwidth_final, fdp_calculate_bandwidth \
      = convert_fdp_to_fp(energy_vec, fdp_vec, bandedge, relativistic_correction)


# Create initial free parameters

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

#### START
energy_vec_base = create_energy_vec(nchannels+1, mean_energy, channel_width, library=np)
params = get_free_params(energy_vec_reference, fdp_vec_reference, energy_vec_base, library=torch)
params_torch = torch.tensor(params)
energy_vec_base_torch = torch.tensor(energy_vec_base)

# From the free parameters, get the physical parameters
fdp_vec, coeff_bandwidth = evaluate_fdp(energy_vec, energy_vec_base_torch, params_torch)

plt.figure()
plt.plot(energy_vec_reference, fdp_vec_reference)
plt.plot(energy_vec, fdp_vec, '.')
plt.xlim([energy_vec_base[0], energy_vec_base[-1]])
plt.savefig("test_params.png")

calculate_fp(energy_vec, energy_vec_base, params, coeff_bandwidth)