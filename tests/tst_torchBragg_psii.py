"""
Simulates a diffraction image of psii with both CCTBX and torchBragg

Usage:
. $MODULES/torchBragg/tests/tst_torchBragg_psii_script.sh
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from exafel_project.kpp_utils.phil import parse_input
from LS49.spectra.generate_spectra import spectra_simulation
from LS49.sim.step4_pad import microcrystal
from torchBragg.kramers_kronig.amplitudes_spread_torch_integration import amplitudes_spread_psii
from torchBragg.kramers_kronig.sf_linearity import get_Fhkl_mat
from torchBragg.forward_simulation.utils import set_basic_params, set_nanoBragg_params, forward_sim_pytorch

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    params,options = parse_input()
    use_background = True
    num_pixels = 128 # change ranges for hkl if this is changed

    if num_pixels == 128:
        direct_algo_res_limit = 10.0 # need to make low enough to include all hkl on detector
        h_max= 11
        h_min= -11
        k_max= 22
        k_min= -22
        l_max= 30
        l_min= -30
    elif num_pixels == 512:
        direct_algo_res_limit = 8.0 # need to make low enough to include all hkl on detector
        h_max= 14
        h_min= -14
        k_max= 27
        k_min= -27
        l_max= 38
        l_min= -38
    elif num_pixels == 3840:
        direct_algo_res_limit = 1.85 # need to make low enough to include all hkl on detector
        h_max= 63
        h_min= -63
        k_max= 120
        k_min= -120
        l_max= 167
        l_min= -167
    else:
        NotImplementedError("num_pixels=%d is not implemented"%num_pixels)

    hkl_ranges = (h_max, h_min, k_max, k_min, l_max, l_min)
    sfall_channels = amplitudes_spread_psii(params, direct_algo_res_limit=direct_algo_res_limit)
    basic_params = set_basic_params(params, sfall_channels)
    num_wavelengths = params.spectrum.nchannels
    Fhkl_mat_vec = get_Fhkl_mat(sfall_channels, num_wavelengths, hkl_ranges, complex_output=False)

    raw_pixels, nanoBragg_params, noise_params, fluence_background = set_nanoBragg_params(params, basic_params, sfall_channels, 
                                                                                          num_pixels, num_pixels, simulate=True, use_background=use_background)


    raw_pixels_pytorch = forward_sim_pytorch(params, basic_params, Fhkl_mat_vec, nanoBragg_params, noise_params, 
                                             fluence_background, hkl_ranges, device, num_pixels, num_pixels, use_background=use_background)


    # raw_pixels, nanoBragg_params, noise_params, fluence_background = tst_one_CPU(params, basic_params, sfall_channels, add_spots, use_background, num_pixels=num_pixels)    
    # raw_pixels_pytorch = tst_one_pytorch(params, basic_params, Fhkl_mat_vec, add_spots, nanoBragg_params, noise_params, fluence_background, use_background, hkl_ranges, num_pixels=num_pixels)

    if use_background:
        plt.figure(); plt.imshow(raw_pixels.as_numpy_array(), vmax=5.0e2, cmap='Greys');plt.colorbar();plt.savefig("raw_pixels.png")
        plt.figure(); plt.imshow(raw_pixels_pytorch.cpu().numpy(), vmax=5.0e2, cmap='Greys');plt.colorbar();plt.savefig("raw_pixels_torch.png")
    else:
        plt.figure(); plt.imshow(raw_pixels.as_numpy_array(), vmax=10e-5, cmap='Greys');plt.colorbar();plt.savefig("raw_pixels.png")
        plt.figure(); plt.imshow(raw_pixels_pytorch.cpu().numpy(), vmax=10e-5, cmap='Greys');plt.colorbar();plt.savefig("raw_pixels_torch.png")
    
    plt.figure(); plt.imshow(raw_pixels.as_numpy_array(), cmap='Greys');plt.colorbar();plt.savefig("raw_pixels_no_cap.png")
    plt.figure(); plt.imshow(raw_pixels_pytorch.cpu().numpy(), cmap='Greys');plt.colorbar();plt.savefig("raw_pixels_torch_no_cap.png")

    
