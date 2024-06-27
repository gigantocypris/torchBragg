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


def get_Fhkl_mat(sfall_channels, num_wavelengths, hkl_ranges, default_F=0, complex_output=True):
    h_max, h_min, k_max, k_min, l_max, l_min = hkl_ranges
    Fhkl_mat_vec = []
    for ind in range(num_wavelengths):
        Fhkl=sfall_channels[ind] # 0th wavelength
        Fhkl_indices = Fhkl._indices.as_vec3_double().as_numpy_array()
        Fhkl_data = Fhkl._data.as_numpy_array()
        Fhkl_indices = [tuple(h) for h in Fhkl_indices]

        Fhkl = {h:v for h,v in zip(Fhkl_indices,Fhkl_data)}
        Fhkl = Fhkl_remove(Fhkl, h_max, h_min, k_max, k_min, l_max, l_min)
        Fhkl_mat = Fhkl_dict_to_mat(Fhkl, h_max, h_min, k_max, k_min, l_max, l_min, default_F, torch, complex_output=complex_output)
        Fhkl_mat_vec.append(Fhkl_mat)
    Fhkl_mat_vec = torch.stack(Fhkl_mat_vec) # shape is (num_wavelengths, 2*h_max+1, 2*k_max+1, 2*l_max+1)
    return Fhkl_mat_vec


def get_reference_structure_factors(params, 
                                    hkl_ranges=(63,-63,120,-120,167,-167),
                                    direct_algo_res_limit=1.85,
                                    MN_labels=["Mn_oxidized_model","Mn_oxidized_model","Mn_reduced_model","Mn_reduced_model"]):

    num_wavelengths = params.spectrum.nchannels
    h_max, h_min, k_max, k_min, l_max, l_min = hkl_ranges

    sfall_channels = amplitudes_spread_psii(params, direct_algo_res_limit=direct_algo_res_limit, MN_labels=MN_labels, complex_output=True)
    Fhkl_mat_vec = get_Fhkl_mat(sfall_channels, num_wavelengths, hkl_ranges)
    return Fhkl_mat_vec

def get_wavelengths(params):
    num_wavelengths = params.spectrum.nchannels
    centerline = float(params.spectrum.nchannels-1)/2.0
    channel_mean_eV = (flex.double(range(params.spectrum.nchannels)) - centerline
                    ) * params.spectrum.channel_width + params.beam.mean_energy
    wavelengths = ENERGY_CONV/channel_mean_eV
    wavelengths = wavelengths.as_numpy_array()
    return wavelengths, num_wavelengths

def get_fp_fdp(wavelengths, num_wavelengths, MN_labels, use_updated_curves=True):

    # need to extrapolate the input data to the wavelength of interest
        
    fp_oxidized_vec = []
    fdp_oxidized_vec = []
    fp_reduced_vec = []
    fdp_reduced_vec = []
    fp_ground_state_vec = []
    fdp_ground_state_vec = []
    fp_0_vec = []
    fdp_0_vec = []

    Mn_ground_state = george_sherrell(full_path("data_sherrell/Mn.dat"))
    Mn_0 = george_sherrell("Mn_fp_0_fdp_0.dat")
    if use_updated_curves:
        Mn_oxidized_model = george_sherrell("MnO2_spliced.dat")
        Mn_reduced_model = george_sherrell("Mn2O3_spliced.dat")
    else:
        Mn_oxidized_model = george_sherrell(full_path("data_sherrell/MnO2_spliced.dat"))
        Mn_reduced_model = george_sherrell(full_path("data_sherrell/Mn2O3_spliced.dat"))

    for ind in range(num_wavelengths):
        fp_oxidized, fdp_oxidized = Mn_oxidized_model.fp_fdp_at_wavelength(wavelengths[ind])
        fp_reduced, fdp_reduced = Mn_reduced_model.fp_fdp_at_wavelength(wavelengths[ind])
        fp_ground_state, fdp_ground_state = Mn_ground_state.fp_fdp_at_wavelength(wavelengths[ind])
        fp_0, fdp_0 = Mn_0.fp_fdp_at_wavelength(wavelengths[ind])

        fp_oxidized_vec.append(fp_oxidized)
        fdp_oxidized_vec.append(fdp_oxidized)
        fp_reduced_vec.append(fp_reduced)
        fdp_reduced_vec.append(fdp_reduced)
        fp_ground_state_vec.append(fp_ground_state)
        fdp_ground_state_vec.append(fdp_ground_state)
        fp_0_vec.append(fp_0)
        fdp_0_vec.append(fdp_0)


    fp_oxidized_vec = torch.tensor(fp_oxidized_vec) # shape is (num_wavelengths)
    fdp_oxidized_vec = torch.tensor(fdp_oxidized_vec) # shape is (num_wavelengths)
    fp_reduced_vec = torch.tensor(fp_reduced_vec) # shape is (num_wavelengths)
    fdp_reduced_vec = torch.tensor(fdp_reduced_vec) # shape is (num_wavelengths)
    fp_ground_state_vec = torch.tensor(fp_ground_state_vec) # shape is (num_wavelengths)
    fdp_ground_state_vec = torch.tensor(fdp_ground_state_vec) # shape is (num_wavelengths)
    fp_0_vec = torch.tensor(fp_0_vec) # shape is (num_wavelengths)
    fdp_0_vec = torch.tensor(fdp_0_vec) # shape is (num_wavelengths)
    
    fp_vec = []
    fdp_vec = []
    for ind,label in enumerate(MN_labels):
        if label == "Mn_oxidized_model":
            fp = fp_oxidized_vec
            fdp = fdp_oxidized_vec
        elif label == "Mn_reduced_model":
            fp = fp_reduced_vec
            fdp = fdp_reduced_vec
        elif label == "Mn_ground_state":
            fp = fp_ground_state_vec
            fdp = fdp_ground_state_vec
        else:
            fp = fp_0_vec
            fdp = fdp_0_vec
        fp_vec.append(fp)
        fdp_vec.append(fdp)
    
    fp_vec = torch.stack(fp_vec) # shape is (num_Mn_atoms, num_wavelengths)
    fdp_vec = torch.stack(fdp_vec) # shape is (num_Mn_atoms, num_wavelengths)

    fp_vec = fp_vec[:,:,None,None,None]
    fdp_vec = fdp_vec[:,:,None,None,None]
    return fp_vec, fdp_vec

def get_base_structure_factors(params, 
                               direct_algo_res_limit=1.85,
                               hkl_ranges=(63,-63,120,-120,167,-167),
                               MN_labels=["Mn_oxidized_model","Mn_oxidized_model","Mn_reduced_model","Mn_reduced_model"]):
    num_wavelengths = params.spectrum.nchannels
    centerline = float(params.spectrum.nchannels-1)/2.0
    channel_mean_eV = (flex.double(range(params.spectrum.nchannels)) - centerline
                    ) * params.spectrum.channel_width + params.beam.mean_energy
    wavelengths = ENERGY_CONV/channel_mean_eV
    wavelengths = wavelengths.as_numpy_array()

    h_max, h_min, k_max, k_min, l_max, l_min = hkl_ranges
    
    Mn_atom_vec = range(4)
    Fhkl_mat_vec_all_Mn_diff = []
    
    for Mn in Mn_atom_vec:
        Fhkl_mat_vec = []

        MN_labels_0 = MN_labels.copy()
        MN_labels_0[Mn] = "Mn_fp_0"
        sfall_channels = amplitudes_spread_psii(params, direct_algo_res_limit=direct_algo_res_limit, MN_labels=MN_labels_0, complex_output=True)
        Fhkl_mat_0 = get_Fhkl_mat(sfall_channels, num_wavelengths, hkl_ranges)

        MN_labels_0[Mn] = "Mn_fp_1"
        sfall_channels = amplitudes_spread_psii(params, direct_algo_res_limit=direct_algo_res_limit, MN_labels=MN_labels_0, complex_output=True)
        Fhkl_mat_1 = get_Fhkl_mat(sfall_channels, num_wavelengths, hkl_ranges)
        Fhkl_mat_diff = Fhkl_mat_1 - Fhkl_mat_0

        Fhkl_mat_vec_all_Mn_diff.append(Fhkl_mat_diff)
    
    Fhkl_mat_vec_all_Mn_diff = torch.stack(Fhkl_mat_vec_all_Mn_diff) # shape is (num_Mn_atoms, num_wavelengths, 2*h_max+1, 2*k_max+1, 2*l_max+1)

    MN_labels_0=["Mn_fp_0_fdp_0","Mn_fp_0_fdp_0","Mn_fp_0_fdp_0","Mn_fp_0_fdp_0"]
    sfall_channels = amplitudes_spread_psii(params, direct_algo_res_limit=direct_algo_res_limit, MN_labels=MN_labels_0, complex_output=True)
    Fhkl_mat_0 = get_Fhkl_mat(sfall_channels, num_wavelengths, hkl_ranges) # shape is (num_wavelengths, 2*h_max+1, 2*k_max+1, 2*l_max+1)
    return Fhkl_mat_0, Fhkl_mat_vec_all_Mn_diff

def construct_structure_factors(fp_vec, fdp_vec, Fhkl_mat_0, Fhkl_mat_vec_all_Mn_diff):

    Fhkl_full = Fhkl_mat_0 + torch.sum((fp_vec + fdp_vec*1j)*Fhkl_mat_vec_all_Mn_diff, axis=0) # shape is (num_wavelengths, 2*h_max+1, 2*k_max+1, 2*l_max+1)

    return Fhkl_full

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

    
    