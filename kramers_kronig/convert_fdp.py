"""
Converts f" to f' by creating a piecewise polynomial fit to f" from 0 to infinity, 
then using direct integration to calculate f' at points of interest (evaluation points).

Note that the boundaries of the piecewise polynomial cannot be at the evalution points for f'
due to the discussion outlined here: https://doi.org/10.1364/OE.22.023628

Usage:
libtbx.python $MODULES/torchBragg/kramers_kronig/convert_fdp.py --prefix [FILENAME PREFIX]
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torchBragg.kramers_kronig.create_fp_fdp_dat_file import full_path, read_dat_file
from torchBragg.kramers_kronig.convert_fdp_helper import convert_fdp_to_fp, create_figures
np.seterr(all='raise')

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

create_figures(energy_vec, fp_vec, fdp_vec, cs_fdp, energy_vec_bandwidth,
               fdp_vec_bandwidth, cs_fdp_bandwidth, energy_vec_0, fdp_vec_0,
               func_fixed_pt_0, popt_0, energy_vec_2, fdp_vec_2, func_fixed_pt_2,
               popt_2, energy_vec_full, fdp_vec_full, shift_0, constant_0, fp_calculate_bandwidth,
               energy_vec_bandwidth_final,
               prefix=prefix)
plt.close()


# Combine the vectors into a single 2D array
combined_array = np.column_stack((energy_vec_bandwidth_final, fp_calculate_bandwidth, fdp_calculate_bandwidth))

# Save the combined array to a .dat file
np.savetxt(prefix + '.dat', combined_array, delimiter='\t', fmt='%0.7f')
print("Saved to " + prefix + ".dat")

# back calculate fdp from fp
cs_fdp_1, energy_vec_bandwidth_1,\
fdp_vec_bandwidth_1, cs_fdp_bandwidth_1, energy_vec_0_1, fdp_vec_0_1,\
func_fixed_pt_0_1, popt_0_1, energy_vec_2_1, fdp_vec_2_1, func_fixed_pt_2_1,\
popt_2_1, energy_vec_full_1, fdp_vec_full_1, shift_0_1, constant_0_1, \
fp_calculate_bandwidth_1, energy_vec_bandwidth_final_1, fdp_calculate_bandwidth_1 \
= convert_fdp_to_fp(energy_vec_bandwidth_final, -fp_calculate_bandwidth, bandedge, relativistic_correction)

plt.figure()
plt.title("Backcalculated fdp")
plt.plot(energy_vec_bandwidth_final_1, fp_calculate_bandwidth_1)
plt.plot(energy_vec_bandwidth_final, fdp_calculate_bandwidth, 'r.')
plt.xlim([energy_vec_bandwidth[0],energy_vec_bandwidth[-1]])
plt.savefig(prefix + "_fdp_back_calculated.png")

create_figures(energy_vec, fdp_vec, -fp_vec, cs_fdp_1, energy_vec_bandwidth_1,
               fdp_vec_bandwidth_1, cs_fdp_bandwidth_1, energy_vec_0_1, fdp_vec_0_1,
               func_fixed_pt_0_1, popt_0_1, energy_vec_2_1, fdp_vec_2_1, func_fixed_pt_2_1,
               popt_2_1, energy_vec_full_1, fdp_vec_full_1, shift_0_1, constant_0_1, fp_calculate_bandwidth_1,
               energy_vec_bandwidth_final_1,
               prefix=prefix+"_backcalculate_fdp")
plt.close()
