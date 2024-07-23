"""
Converts f" to f' by creating a piecewise polynomial fit to f" from 0 to infinity, 
then using direct integration to calculate f' at points of interest (evaluation points).

Note that the boundaries of the piecewise polynomial cannot be at the evalution points for f'
due to the discussion outlined here: https://doi.org/10.1364/OE.22.023628

Usage:
source my_env
cd $WORKING_DIR
libtbx.python $MODULES/torchBragg/kramers_kronig/convert_fdp.py --prefix [FILENAME PREFIX]
"""
import time
import argparse
import numpy as np
import torch
from torchBragg.kramers_kronig.create_fp_fdp_dat_file import full_path, read_dat_file
from torchBragg.kramers_kronig.convert_fdp_helper import create_energy_vec, check_clashes, get_free_params, get_coeff_bandwidth, \
    get_physical_params_fdp, reformat_fdp, get_reformatted_fdp
from torchBragg.kramers_kronig.convert_fdp_helper_vectorize import get_physical_params_fp
from torchBragg.kramers_kronig.convert_fdp_visualizer import create_figures

torch.set_default_dtype(torch.float64)
np.seterr(all='raise')

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
    print('Numerical check of reformatted coefficients:', torch.allclose(fdp_final, fdp_check, rtol=1e-05, atol=1e-08, equal_nan=False))

    return fdp_final, fp_final, fdp_check

if __name__ == '__main__':
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
    channel_width = 1
    nchannels = 100

    energy_vec_reference, fp_vec_reference, fdp_vec_reference = read_dat_file(Mn_model)
    energy_vec_reference = np.array(energy_vec_reference).astype(np.float64)
    fp_vec_reference = np.array(fp_vec_reference).astype(np.float64)
    fdp_vec_reference = np.array(fdp_vec_reference).astype(np.float64)

    # points where the FREE PARAMTERS are located
    energy_vec_bandwidth = create_energy_vec(nchannels+1, bandedge, channel_width, library=np)

    # points where the PHYSICAL PARAMETERS are located (cannot be the same as free parameter locations)
    energy_vec_final = torch.tensor(energy_vec_reference)
    # energy_vec_final = create_energy_vec(nchannels, bandedge, channel_width, library=torch) # smaller vector for optimization

    # energy_vec_bandwidth cannot have the same values as energy_vec_final
    if check_clashes(energy_vec_bandwidth, energy_vec_final)>0:
        raise Exception("Matching values in energy_vec_bandwidth and energy_vec_final, remove clashes")

    start_time = time.time()
    fdp_final, fp_final, fdp_check = convert_fdp_to_fp(energy_vec_reference, energy_vec_bandwidth, energy_vec_final, 
                                            fdp_vec_reference, relativistic_correction)
    end_time = time.time()

    print('Total time: ', end_time-start_time)
    
    # plots
    create_figures(energy_vec_reference, energy_vec_bandwidth, fp_vec_reference, fdp_vec_reference, energy_vec_final, \
                   fdp_final, fp_final, fdp_check, prefix=prefix)

    # Combine the vectors into a single 2D array
    combined_array = np.column_stack((energy_vec_final, fp_final, fdp_final))

    # Save the combined array to a .dat file
    np.savetxt(prefix + '.dat', combined_array, delimiter='\t', fmt='%0.7f')
    print("Saved to " + prefix + ".dat")