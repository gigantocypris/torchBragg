import os
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from exafel_project.kpp_utils.phil import parse_input
from LS49.spectra.generate_spectra import spectra_simulation
from LS49.sim.step4_pad import microcrystal
from LS49 import ls49_big_data, legacy_random_orientations
from LS49.sim.fdp_plot import george_sherrell
from exafel_project.kpp_utils.ferredoxin import basic_detector_rayonix
from torchBragg.amplitudes_spread_torchBragg import amplitudes_spread_psii
from simtbx.nanoBragg import nanoBragg
from simtbx.nanoBragg import shapetype
from scitbx.array_family import flex
import scitbx
from scitbx.matrix import sqr,col
from diffraction_vectorized import add_torchBragg_spots
from add_background_vectorized import add_background
from utils_vectorized import Fhkl_remove, Fhkl_dict_to_mat
from add_noise import add_noise
from create_fp_fdp_dat_file import full_path
from scipy import constants
from torchBragg.kramers_kronig.sf_linearity import get_reference_structure_factors, get_wavelengths, get_fp_fdp, get_base_structure_factors, construct_structure_factors
ENERGY_CONV = 1e10*constants.c*constants.h / constants.electron_volt
torch.set_default_dtype(torch.float64)


if __name__ == "__main__":
    params,options = parse_input()

    hkl_ranges=(11, -11, 22, -22, 30, -30)
    direct_algo_res_limit=10.0
    MN_labels=["Mn_oxidized_model","Mn_oxidized_model","Mn_reduced_model","Mn_reduced_model"]
    
    Fhkl_mat_vec_0 = get_reference_structure_factors(params, hkl_ranges=hkl_ranges, direct_algo_res_limit=direct_algo_res_limit, MN_labels=MN_labels)
    
    wavelengths, num_wavelengths = get_wavelengths(params)
    fp_vec, fdp_vec = get_fp_fdp(wavelengths,num_wavelengths, MN_labels)
    Fhkl_mat_0, Fhkl_mat_vec_all_Mn_diff = get_base_structure_factors(params, direct_algo_res_limit=direct_algo_res_limit, hkl_ranges=hkl_ranges, MN_labels=MN_labels)
    Fhkl_mat_vec_1 = construct_structure_factors(fp_vec, fdp_vec, Fhkl_mat_0, Fhkl_mat_vec_all_Mn_diff)

    print(Fhkl_mat_vec_0[torch.abs(Fhkl_mat_vec_0)>=1e-10])
    print(Fhkl_mat_vec_1[torch.abs(Fhkl_mat_vec_0)>=1e-10])

    print("Differences:")
    print(Fhkl_mat_vec_1[torch.abs(Fhkl_mat_vec_0)>=1e-10]-Fhkl_mat_vec_0[torch.abs(Fhkl_mat_vec_0)>=1e-10])
    assert torch.allclose(Fhkl_mat_vec_0[torch.abs(Fhkl_mat_vec_0)>=1e-10], Fhkl_mat_vec_1[torch.abs(Fhkl_mat_vec_0)>=1e-10], atol=2e-1)

    
    