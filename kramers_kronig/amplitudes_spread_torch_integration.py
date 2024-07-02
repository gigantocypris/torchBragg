"""
Functions to get the ground truth structure factors accounting for the anomalous scattering.
This is modified from amplitudes_spread.py in CCTBX by including a framework to zero out fp and fdp.
This aids in the optimization of fp and fdp by allowing easy construction of modified structure factors
without dependency on the non-differentiable forward physics of CCTBX.
"""

from __future__ import division, print_function
import os

from LS49 import ls49_big_data
from scitbx.array_family import flex
from LS49.sim.util_fmodel import gen_fmodel
from exafel_project.kpp_utils.ferredoxin import data
import exafel_project.kpp_utils.amplitudes_spread_psii as psii_fn
from torchBragg.kramers_kronig.create_fp_fdp_dat_file import full_path

from scipy import constants
ENERGY_CONV = 1e10*constants.c*constants.h / constants.electron_volt

def amplitudes_spread_ferredoxin(params, direct_algo_res_limit=1.7):

  wavelength_A = ENERGY_CONV / params.beam.mean_energy
  # general ballpark X-ray wavelength in Angstroms, does not vary shot-to-shot
  centerline = float(params.spectrum.nchannels-1)/2.0
  channel_mean_eV = (flex.double(range(params.spectrum.nchannels)) - centerline
                      ) * params.spectrum.channel_width + params.beam.mean_energy
  wavelengths = ENERGY_CONV/channel_mean_eV

  local_data = data() # later put this through broadcast

  # this is PDB 1M2A
  GF = gen_fmodel(resolution=direct_algo_res_limit,
                  pdb_text=local_data.get("pdb_lines"),algorithm="fft",wavelength=wavelength_A)
  GF.set_k_sol(0.435)
  GF.make_P1_primitive()

  # Generating sf for my wavelengths
  sfall_channels = {}

  for x in range(len(wavelengths)):

    GF.reset_wavelength(wavelengths[x])
    GF.reset_specific_at_wavelength(
                     label_has="FE1",tables=local_data.get("Fe_oxidized_model"),newvalue=wavelengths[x])
    GF.reset_specific_at_wavelength(
                     label_has="FE2",tables=local_data.get("Fe_reduced_model"),newvalue=wavelengths[x])
    sfall_channels[x]=GF.get_amplitudes()

  return sfall_channels



def psii_data():
  from LS49.sim.fdp_plot import george_sherrell
  return dict(
    pdb_lines=open(full_path("7RF1_refine_030_Aa_refine_032_refine_034.pdb"), "r").read(),
    Mn_oxidized_model=george_sherrell(full_path("data_sherrell/MnO2_spliced.dat")),
    Mn_reduced_model=george_sherrell(full_path("data_sherrell/Mn2O3_spliced.dat")),
    Mn_metallic_model=george_sherrell(full_path("data_sherrell/Mn.dat")),
    Mn_fp_0=george_sherrell("Mn_fp_0_fdp_-.dat"),
    Mn_fp_1=george_sherrell("Mn_fp_1_fdp_-.dat"),
    Mn_fdp_0=george_sherrell("Mn_fp_-_fdp_0.dat"),
    Mn_fdp_1=george_sherrell("Mn_fp_-_fdp_1.dat"),
    Mn_fp_0_fdp_0=george_sherrell("Mn_fp_0_fdp_0.dat"),

  )

def amplitudes_spread_psii(params, direct_algo_res_limit=1.85, 
                           MN_labels=["Mn_oxidized_model","Mn_oxidized_model","Mn_reduced_model","Mn_reduced_model"],
                           complex_output=False):
  wavelength_A = ENERGY_CONV / params.beam.mean_energy
  # general ballpark X-ray wavelength in Angstroms, does not vary shot-to-shot
  centerline = float(params.spectrum.nchannels-1)/2.0
  channel_mean_eV = (flex.double(range(params.spectrum.nchannels)) - centerline
                      ) * params.spectrum.channel_width + params.beam.mean_energy
  wavelengths = ENERGY_CONV/channel_mean_eV

  local_data = psii_data()  # later put this through broadcast

  # this is PDB 7RF1_refine_030_Aa_refine_032_refine_034
  GF = gen_fmodel(resolution=direct_algo_res_limit,
                  pdb_text=local_data.get("pdb_lines"),
                  algorithm="fft", wavelength=wavelength_A)
  GF.set_k_sol(0.435)
  GF.make_P1_primitive()

  # Generating sf for my wavelengths
  sfall_channels = {}

  for x in range(len(wavelengths)):
    GF.reset_wavelength(wavelengths[x])  # XXX TODO: which to make 3+ and which 4+?
    GF.reset_specific_at_wavelength(label_has="MN1",
                                    tables=local_data.get(MN_labels[0]),
                                    newvalue=wavelengths[x])
    GF.reset_specific_at_wavelength(label_has="MN2",
                                    tables=local_data.get(MN_labels[1]),
                                    newvalue=wavelengths[x])
    GF.reset_specific_at_wavelength(label_has="MN3",
                                    tables=local_data.get(MN_labels[2]),
                                    newvalue=wavelengths[x])
    GF.reset_specific_at_wavelength(label_has="MN4",
                                    tables=local_data.get(MN_labels[3]),
                                    newvalue=wavelengths[x])

    if complex_output:
      sfall_channels[x] = GF.get_fmodel().f_model 
    else:
      sfall_channels[x] = GF.get_amplitudes()

  return sfall_channels
