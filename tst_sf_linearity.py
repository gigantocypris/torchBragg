import os
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from exafel_project.kpp_utils.phil import parse_input
from LS49.spectra.generate_spectra import spectra_simulation
from LS49.sim.step4_pad import microcrystal
from LS49 import ls49_big_data, legacy_random_orientations
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
torch.set_default_dtype(torch.float64)


def set_basic_params(params, direct_algo_res_limit):
    spectra = spectra_simulation()
    iterator = spectra.generate_recast_renormalized_image_parameterized(image=0,params=params)
    wavlen, flux, shot_to_shot_wavelength_A = next(iterator) # list of lambdas, list of fluxes, average wavelength

    MN_labels_0=["Mn_oxidized_model","Mn_oxidized_model","Mn_reduced_model","Mn_reduced_model"]
    sfall_channels_0 = amplitudes_spread_psii(params, direct_algo_res_limit=direct_algo_res_limit, MN_labels=MN_labels_0)

    # STOPPED HERE
    
    """
    Fhkl=sfall_channels_0[0] # 0th wavelength
    Fhkl_indices = Fhkl._indices.as_vec3_double().as_numpy_array()
    Fhkl_data = Fhkl._data.as_numpy_array()
    Fhkl_indices = [tuple(h) for h in Fhkl_indices]

    Fhkl = {h:v for h,v in zip(Fhkl_indices,Fhkl_data)}
    Fhkl = Fhkl_remove(Fhkl, h_max, h_min, k_max, k_min, l_max, l_min)
    Fhkl_mat = Fhkl_dict_to_mat(Fhkl, h_max, h_min, k_max, k_min, l_max, l_min, default_F, torch)
    """
    breakpoint()

    Mn_atom_vec = range(4)
    sfall_channels_vec = []

    for Mn in Mn_atom_vec:
        MN_labels = MN_labels_0
        MN_labels[Mn] = "Mn_fp_0"
        sfall_channels = amplitudes_spread_psii(params, direct_algo_res_limit=direct_algo_res_limit, MN_labels=MN_labels_0)
        sfall_channels_vec.append(sfall_channels)

        MN_labels[Mn] = "Mn_fp_1"
        sfall_channels = amplitudes_spread_psii(params, direct_algo_res_limit=direct_algo_res_limit, MN_labels=MN_labels_0)
        sfall_channels_vec.append(sfall_channels)

        MN_labels[Mn] = "Mn_fdp_0"
        sfall_channels = amplitudes_spread_psii(params, direct_algo_res_limit=direct_algo_res_limit, MN_labels=MN_labels_0)
        sfall_channels_vec.append(sfall_channels)

        MN_labels[Mn] = "Mn_fdp_1"
        sfall_channels = amplitudes_spread_psii(params, direct_algo_res_limit=direct_algo_res_limit, MN_labels=MN_labels_0)
        sfall_channels_vec.append(sfall_channels)
    
    # re-create the original sfall_channels at a given wavelength
    """
    Mn_oxidized_model = full_path("data_sherrell/MnO2_spliced.dat")
    Mn_reduced_model = full_path("data_sherrell/Mn2O3_spliced.dat")
    """
    # need to extrapolate the input data to the wavelength of interest

    return sfall_channels

if __name__ == "__main__":
    params,options = parse_input()
    direct_algo_res_limit = 1.85
    sfall_channels = set_basic_params(params, direct_algo_res_limit)

    Mn_oxidized_model = full_path("data_sherrell/MnO2_spliced.dat")
    Mn_reduced_model = full_path("data_sherrell/Mn2O3_spliced.dat")