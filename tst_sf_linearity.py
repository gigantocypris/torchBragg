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
ENERGY_CONV = 1e10*constants.c*constants.h / constants.electron_volt
torch.set_default_dtype(torch.float64)


def get_Fhkl_mat(sfall_channels, num_wavelengths, hkl_limits, default_F=0):
    h_max, h_min, k_max, k_min, l_max, l_min = hkl_limits
    Fhkl_mat_vec = []
    for ind in range(num_wavelengths):
        Fhkl=sfall_channels[ind] # 0th wavelength
        Fhkl_indices = Fhkl._indices.as_vec3_double().as_numpy_array()
        Fhkl_data = Fhkl._data.as_numpy_array()
        Fhkl_indices = [tuple(h) for h in Fhkl_indices]

        Fhkl = {h:v for h,v in zip(Fhkl_indices,Fhkl_data)}
        Fhkl = Fhkl_remove(Fhkl, h_max, h_min, k_max, k_min, l_max, l_min)
        Fhkl_mat = Fhkl_dict_to_mat(Fhkl, h_max, h_min, k_max, k_min, l_max, l_min, default_F, torch)
        Fhkl_mat_vec.append(Fhkl_mat)
        breakpoint()
    return Fhkl_mat_vec




def get_reference_structure_factors(params, 
                                    hkl_limits=(63,-63,120,-120,167,-167),
                                    direct_algo_res_limit=1.85,
                                    MN_labels=["Mn_oxidized_model","Mn_oxidized_model","Mn_reduced_model","Mn_reduced_model"]):

    num_wavelengths = params.spectrum.nchannels
    h_max, h_min, k_max, k_min, l_max, l_min = hkl_limits

    sfall_channels = amplitudes_spread_psii(params, direct_algo_res_limit=direct_algo_res_limit, MN_labels=MN_labels)
    Fhkl_mat_vec = get_Fhkl_mat(sfall_channels, num_wavelengths, hkl_limits)

    return Fhkl_mat_vec


def construct_structure_factors(params, 
                                hkl_limits=(63,-63,120,-120,167,-167),
                                MN_labels=["Mn_oxidized_model","Mn_oxidized_model","Mn_reduced_model","Mn_reduced_model"]):

    num_wavelengths = params.spectrum.nchannels
    centerline = float(params.spectrum.nchannels-1)/2.0
    channel_mean_eV = (flex.double(range(params.spectrum.nchannels)) - centerline
                    ) * params.spectrum.channel_width + params.beam.mean_energy
    wavelengths = ENERGY_CONV/channel_mean_eV
    wavelengths = wavelengths.as_numpy_array()

    h_max, h_min, k_max, k_min, l_max, l_min = hkl_limits
    
    Mn_atom_vec = range(4)
    Fhkl_mat_vec_all_Mn = []

    # STOPPED HERE
    # Think through logic, may need to change create_fp_fdp_dat_file.py
    # Make everything into a tensor, no Python lists
    # Tensor operations to get the final values
    
    for Mn in Mn_atom_vec:
        Fhkl_mat_vec = []

        MN_labels_0 = MN_labels.copy()
        MN_labels_0[Mn] = "Mn_fp_0"
        sfall_channels = amplitudes_spread_psii(params, direct_algo_res_limit=direct_algo_res_limit, MN_labels=MN_labels_0)
        Fhkl_mat_vec.append(get_Fhkl_mat(sfall_channels, num_wavelengths, hkl_limits))

        MN_labels_0[Mn] = "Mn_fp_1"
        sfall_channels = amplitudes_spread_psii(params, direct_algo_res_limit=direct_algo_res_limit, MN_labels=MN_labels_0)
        Fhkl_mat_vec.append(get_Fhkl_mat(sfall_channels, num_wavelengths, hkl_limits))

        MN_labels_0[Mn] = "Mn_fdp_0"
        sfall_channels = amplitudes_spread_psii(params, direct_algo_res_limit=direct_algo_res_limit, MN_labels=MN_labels_0)
        Fhkl_mat_vec.append(get_Fhkl_mat(sfall_channels, num_wavelengths, hkl_limits))

        MN_labels_0[Mn] = "Mn_fdp_1"
        sfall_channels = amplitudes_spread_psii(params, direct_algo_res_limit=direct_algo_res_limit, MN_labels=MN_labels_0)
        Fhkl_mat_vec.append(get_Fhkl_mat(sfall_channels, num_wavelengths, hkl_limits))

        Fhkl_mat_vec_all_Mn.append(Fhkl_mat_vec)
    
    # need to extrapolate the input data to the wavelength of interest
        
    fp_oxidized_vec = []
    fdp_oxidized_vec = []
    fp_reduced_vec = []
    fdp_reduced_vec = []

    Mn_oxidized_model = george_sherrell(full_path("data_sherrell/MnO2_spliced.dat"))
    Mn_reduced_model = george_sherrell(full_path("data_sherrell/Mn2O3_spliced.dat"))

    for ind in range(num_wavelengths):
        fp_oxidized, fdp_oxidized = Mn_oxidized_model.fp_fdp_at_wavelength(wavelengths[ind])
        fp_reduced, fdp_reduced = Mn_reduced_model.fp_fdp_at_wavelength(wavelengths[ind])

        fp_oxidized_vec.append(fp_oxidized)
        fdp_oxidized_vec.append(fdp_oxidized)
        fp_reduced_vec.append(fp_reduced)
        fdp_reduced_vec.append(fdp_reduced)

    # re-create the original sfall_channels
        
    breakpoint()

    for label in MN_labels:
        if label == "Mn_oxidized_model":
            fp_oxidized_vec
            fdp_oxidized_vec
        elif label == "Mn_reduced_model":
            fp_reduced_vec
            fdp_reduced_vec
        else:
            NotImplementedError("Only Mn_oxidized_model and Mn_reduced_model are supported")

    return sfall_channels

if __name__ == "__main__":
    params,options = parse_input()

    hkl_limits=(11, -11, 22, -22, 30, -30)
    direct_algo_res_limit=10.0
    MN_labels=["Mn_oxidized_model","Mn_oxidized_model","Mn_reduced_model","Mn_reduced_model"]
    
    Fhkl_mat_vec_0 = get_reference_structure_factors(params, hkl_limits=hkl_limits, direct_algo_res_limit=direct_algo_res_limit, MN_labels=MN_labels)
    Fhkl_mat_vec_1 = construct_structure_factors(params, hkl_limits=hkl_limits, MN_labels=MN_labels)

    
    