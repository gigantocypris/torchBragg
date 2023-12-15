from __future__ import division, print_function
from scitbx.array_family import flex

from LS49.sim.util_fmodel import gen_fmodel
from exafel_project.kpp_utils.ferredoxin import data

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

