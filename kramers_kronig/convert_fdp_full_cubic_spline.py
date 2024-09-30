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
from torchBragg.kramers_kronig.convert_fdp_full_cubic_spline_helper import get_free_params, get_coeff_cubic_spline, \
    reformat_fdp, get_physical_params_fdp
from torchBragg.kramers_kronig.convert_fdp_helper import create_energy_vec, check_clashes
from torchBragg.kramers_kronig.convert_fdp_helper_vectorize import fdp_fp_integrate
from torchBragg.kramers_kronig.convert_fdp_visualizer import create_figures_full_cubic_spline

torch.set_default_dtype(torch.float64)
np.seterr(all='raise')


def convert_fdp_to_fp(energy_vec_reference, energy_vec_free, energy_vec_physical, fdp_vec_reference, relativistic_correction, device='cpu'):
    """ 
    energy_vec_reference is the x-values for fdp_vec_reference
    energy_vec_free denotes the x-values of the points we in the bandwidth that define the cubic spline fit there
    energy_vec_physical are the x-values for the physical parameters fdp and fp 
    
    energy_vec_free cannot have any points in energy_vec_reference
    """
    fdp_vec_free = get_free_params(energy_vec_reference, fdp_vec_reference, energy_vec_free)
    
    fdp_vec_free = torch.tensor(fdp_vec_free).to(device)

    energy_vec_free = torch.tensor(energy_vec_free).to(device)
    params_free_cubic_spline = get_coeff_cubic_spline(energy_vec_free, fdp_vec_free)
    fdp_vec_physical = get_physical_params_fdp(energy_vec_physical, energy_vec_free, params_free_cubic_spline)

    intervals_mat, coeff_mat, powers_mat = reformat_fdp(energy_vec_free, params_free_cubic_spline, device=device)
    fp_vec_physical = fdp_fp_integrate(energy_vec_physical, intervals_mat, coeff_mat, powers_mat, relativistic_correction, device=device)

    return fp_vec_physical, fdp_vec_physical

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, default='MnO2_spliced',
                        help='filename prefix for the fp and fdp curves to load',
                        choices=['Fe', 'Mn', 'MnO2_spliced', 'Mn2O3_spliced'])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # points where the FREE PARAMETERS are located
    energy_vec_bandwidth = create_energy_vec(nchannels+1, bandedge, channel_width, library=np)
    energy_vec_free = np.concatenate((np.array([1000.]), energy_vec_bandwidth, np.array([20000.])))

    # points where the PHYSICAL PARAMETERS are located (cannot be the same as free parameter locations)
    energy_vec_physical = create_energy_vec(nchannels, bandedge, channel_width, library=torch).to(device)

    # energy_vec_bandwidth cannot have the same values as energy_vec_final
    if check_clashes(energy_vec_free, energy_vec_physical.cpu().numpy())>0:
        raise Exception("Matching values in energy_vec_bandwidth and energy_vec_final, remove clashes")

    start_time = time.time()
    fp_vec_physical, fdp_vec_physical = convert_fdp_to_fp(energy_vec_reference, energy_vec_free, energy_vec_physical, 
                                            fdp_vec_reference, relativistic_correction, device=device)
    end_time = time.time()

    print('Total time: ', end_time-start_time)
    
    # plots
    create_figures_full_cubic_spline(energy_vec_reference, fp_vec_reference, fdp_vec_reference, energy_vec_physical.cpu().numpy(), fp_vec_physical.cpu().numpy(), fdp_vec_physical.cpu().numpy(), bandedge, prefix=prefix + '_full_cubic')
