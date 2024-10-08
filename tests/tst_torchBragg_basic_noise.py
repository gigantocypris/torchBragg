"""
Same as tst_nanoBragg_basic.py, but without noise, only background
"""

from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import torch
import numpy as np
import time
from torchBragg.forward_simulation.naive.utils import which_package
from scitbx.array_family import flex
from simtbx.nanoBragg import testuple
from simtbx.nanoBragg import shapetype
from simtbx.nanoBragg import convention
from simtbx.nanoBragg import nanoBragg
import libtbx.load_env # possibly implicit
from cctbx import crystal
from cctbx import miller
assert miller
from torchBragg.forward_simulation.vectorized.utils_vectorized import Fhkl_remove, Fhkl_dict_to_mat
from torchBragg.forward_simulation.vectorized.add_noise import add_noise
from torchBragg.forward_simulation.pdb_helper import fcalc_from_pdb

torch.set_default_dtype(torch.float64)

pdb_lines = """HEADER TEST
CRYST1   50.000   60.000   70.000  90.00  90.00  90.00 P 1
ATOM      1  O   HOH A   1      56.829   2.920  55.702  1.00 20.00           O
ATOM      2  O   HOH A   2      49.515  35.149  37.665  1.00 20.00           O
ATOM      3  O   HOH A   3      52.667  17.794  69.925  1.00 20.00           O
ATOM      4  O   HOH A   4      40.986  20.409  18.309  1.00 20.00           O
ATOM      5  O   HOH A   5      46.896  37.790  41.629  1.00 20.00           O
ATOM      6 SED  MSE A   6       1.000   2.000   3.000  1.00 20.00          SE
END
"""

def tst_nanoBragg_basic(spixels, fpixels, add_background_bool, add_noise_bool):
  SIM = nanoBragg(detpixels_slowfast=(spixels,fpixels),pixel_size_mm=0.1,Ncells_abc=(5,5,5),verbose=9)
  SIM.seed = 10
  SIM.randomize_orientation()
  SIM.distance_mm=100

  SIM.oversample=1
  SIM.wavelength_A=1
  SIM.polarization=1
  #SIM.unit_cell_tuple=(50,50,50,90,90,90)
  print("unit_cell_Adeg=",SIM.unit_cell_Adeg)
  print("unit_cell_tuple=",SIM.unit_cell_tuple)
  # this will become F000, marking the beam center
  SIM.F000=200
  SIM.default_F=0
  #SIM.missets_deg= (10,20,30)
  print("mosaic_seed=",SIM.mosaic_seed)
  print("seed=",SIM.seed)
  print("calib_seed=",SIM.calib_seed)
  print("missets_deg =", SIM.missets_deg)
  sfall = fcalc_from_pdb(resolution=1.6,pdb_lines=pdb_lines,algorithm="direct",wavelength=SIM.wavelength_A)
  # use crystal structure to initialize Fhkl array
  SIM.Fhkl=sfall
  # fastest option, least realistic
  SIM.xtal_shape=shapetype.Tophat
  # only really useful for long runs
  SIM.progress_meter=False
  # prints out value of one pixel only.  will not render full image!
  #SIM.printout_pixel_fastslow=(500,500)
  #SIM.printout=True
  SIM.show_params()
  # flux is always in photons/s
  SIM.flux=1e12
  # assumes round beam
  SIM.beamsize_mm=0.1
  SIM.exposure_s=0.1
  print("Ncells_abc=",SIM.Ncells_abc)
  print("xtal_size_mm=",SIM.xtal_size_mm)
  print("unit_cell_Adeg=",SIM.unit_cell_Adeg)
  print("unit_cell_tuple=",SIM.unit_cell_tuple)
  print("missets_deg=",SIM.missets_deg)
  print("Amatrix=",SIM.Amatrix)
  #SIM.beamcenter_convention=convention.ADXV
  #SIM.beam_center_mm=(45,47)
  print("beam_center_mm=",SIM.beam_center_mm)
  print("XDS_ORGXY=",SIM.XDS_ORGXY)
  print("detector_pivot=",SIM.detector_pivot)
  print("xtal_shape=",SIM.xtal_shape)
  print("beamcenter_convention=",SIM.beamcenter_convention)
  print("fdet_vector=",SIM.fdet_vector)
  print("sdet_vector=",SIM.sdet_vector)
  print("odet_vector=",SIM.odet_vector)
  print("beam_vector=",SIM.beam_vector)
  print("polar_vector=",SIM.polar_vector)
  print("spindle_axis=",SIM.spindle_axis)
  print("twotheta_axis=",SIM.twotheta_axis)
  print("distance_meters=",SIM.distance_meters)
  print("distance_mm=",SIM.distance_mm)
  print("close_distance_mm=",SIM.close_distance_mm)
  print("detector_twotheta_deg=",SIM.detector_twotheta_deg)
  print("detsize_fastslow_mm=",SIM.detsize_fastslow_mm)
  print("detpixels_fastslow=",SIM.detpixels_fastslow)
  print("detector_rot_deg=",SIM.detector_rot_deg)
  print("curved_detector=",SIM.curved_detector)
  print("pixel_size_mm=",SIM.pixel_size_mm)
  print("point_pixel=",SIM.point_pixel)
  print("polarization=",SIM.polarization)
  print("nopolar=",SIM.nopolar)
  print("oversample=",SIM.oversample)
  print("region_of_interest=",SIM.region_of_interest)
  print("wavelength_A=",SIM.wavelength_A)
  print("energy_eV=",SIM.energy_eV)
  print("fluence=",SIM.fluence)
  print("flux=",SIM.flux)
  print("exposure_s=",SIM.exposure_s)
  print("beamsize_mm=",SIM.beamsize_mm)
  print("dispersion_pct=",SIM.dispersion_pct)
  print("dispsteps=",SIM.dispsteps)
  print("divergence_hv_mrad=",SIM.divergence_hv_mrad)
  print("divsteps_hv=",SIM.divsteps_hv)
  print("divstep_hv_mrad=",SIM.divstep_hv_mrad)
  print("round_div=",SIM.round_div)
  print("phi_deg=",SIM.phi_deg)
  print("osc_deg=",SIM.osc_deg)
  print("phisteps=",SIM.phisteps)
  print("phistep_deg=",SIM.phistep_deg)
  print("detector_thick_mm=",SIM.detector_thick_mm)
  print("detector_thicksteps=",SIM.detector_thicksteps)
  print("detector_thickstep_mm=",SIM.detector_thickstep_mm)
  print("mosaic_spread_deg=",SIM.mosaic_spread_deg)
  print("mosaic_domains=",SIM.mosaic_domains)
  print("indices=",SIM.indices)
  print("amplitudes=",SIM.amplitudes)
  print("Fhkl_tuple=",SIM.Fhkl_tuple)
  print("default_F=",SIM.default_F)
  print("interpolate=",SIM.interpolate)
  print("integral_form=",SIM.integral_form)
  # now actually burn up some CPU
  SIM.show_params()
#   SIM.add_nanoBragg_spots_cuda()
  SIM.add_nanoBragg_spots()

  params = (SIM.phisteps, SIM.mosaic_domains, SIM.oversample, SIM.pixel_size_mm, SIM.detector_thicksteps,
            SIM.spot_scale, SIM.fluence, SIM.detector_thickstep_mm, SIM.fdet_vector, SIM.sdet_vector, SIM.odet_vector,
            SIM.pix0_vector_mm, SIM.curved_detector, SIM.distance_mm, SIM.beam_vector, SIM.close_distance_mm,
            SIM.point_pixel, SIM.detector_thick_mm, SIM.Ncells_abc, SIM.integral_form, 
            SIM.Fhkl._indices.as_vec3_double().as_numpy_array(), SIM.Fhkl._data.as_numpy_array(), SIM.default_F,
            SIM.nopolar, SIM.polarization, SIM.polar_vector, SIM.verbose,
            )
  # simulated crystal is only 125 unit cells (25 nm wide)
  # amplify spot signal to simulate physical crystal of 4000x larger: 100 um (64e9 x the volume)

  SIM.raw_pixels *= 64e9

  if add_background_bool:
    # rough approximation to water: interpolation points for sin(theta/lambda) vs structure factor
    bg = flex.vec2_double([(0,2.57),(0.0365,2.58),(0.07,2.8),(0.12,5),(0.162,8),(0.2,6.75),(0.18,7.32),(0.216,6.75),(0.236,6.5),(0.28,4.5),(0.3,4.3),(0.345,4.36),(0.436,3.77),(0.5,3.17)])
    SIM.Fbg_vs_stol = bg
    SIM.amorphous_sample_thick_mm = 0.1
    SIM.amorphous_density_gcm3 = 1
    SIM.amorphous_molecular_weight_Da = 18
    SIM.flux=1e12
    SIM.beamsize_mm=0.1
    SIM.exposure_s=0.1
    SIM.add_background()

    # rough approximation to air
    bg = flex.vec2_double([(0,14.1),(0.045,13.5),(0.174,8.35),(0.35,4.78),(0.5,4.22)])
    SIM.Fbg_vs_stol = bg
    SIM.amorphous_sample_thick_mm = 35 # between beamstop and collimator
    SIM.amorphous_density_gcm3 = 1.2e-3
    SIM.amorphous_sample_molecular_weight_Da = 28 # nitrogen = N2
    print("amorphous_sample_size_mm=",SIM.amorphous_sample_size_mm)
    print("amorphous_sample_thick_mm=",SIM.amorphous_sample_thick_mm)
    print("amorphous_density_gcm3=",SIM.amorphous_density_gcm3)
    print("amorphous_molecular_weight_Da=",SIM.amorphous_molecular_weight_Da)
    SIM.add_background()
  
  if add_noise_bool:
    # set this to 0 or -1 to trigger automatic radius.  could be very slow with bright images
    SIM.detector_psf_kernel_radius_pixels=5
    SIM.detector_psf_fwhm_mm=0.08
    SIM.detector_psf_type=shapetype.Fiber
    print(SIM.raw_pixels[200])

    print("quantum_gain=",SIM.quantum_gain)
    print("adc_offset_adu=",SIM.adc_offset_adu)
    print("detector_calibration_noise_pct=",SIM.detector_calibration_noise_pct)
    print("flicker_noise_pct=",SIM.flicker_noise_pct)
    print("readout_noise_adu=",SIM.readout_noise_adu)
    print("detector_psf_type=",SIM.detector_psf_type)
    print("detector_psf_fwhm_mm=",SIM.detector_psf_fwhm_mm)
    print("detector_psf_kernel_radius_pixels=",SIM.detector_psf_kernel_radius_pixels)

    SIM.add_noise()    
    noise_params = (SIM.quantum_gain, SIM.adc_offset_adu, SIM.detector_calibration_noise_pct, 
                    SIM.flicker_noise_pct, SIM.readout_noise_adu, SIM.detector_psf_type, 
                    SIM.detector_psf_fwhm_mm, SIM.detector_psf_kernel_radius_pixels)
  else:
    noise_params = None
  print("Value of pixel: ",SIM.raw_pixels[200])

  return(SIM.raw_pixels, params, noise_params)

def convert_vector(tuple, use_numpy):
  prefix, new_array = which_package(use_numpy)
  return(new_array([0, tuple[0],tuple[1],tuple[2]]))


def tst_torchBragg_basic(spixels, fpixels, params, use_numpy, vectorize, add_background_bool, noise_params, convolution_type='real_space'):
  prefix, new_array = which_package(use_numpy)
  phisteps, mosaic_domains, oversample, pixel_size_mm, detector_thicksteps, spot_scale, fluence, \
  detector_thickstep_mm, fdet_vector, sdet_vector, odet_vector, pix0_vector_mm, curved_detector, \
  distance_mm, beam_vector, close_distance_mm, point_pixel, detector_thick_mm, Ncells_abc, \
  integral_form, Fhkl_indices, Fhkl_data, default_F, nopolar, polarization, polar_vector, verbose = params


  pixel_size = pixel_size_mm/1000
  roi_xmin = 0 
  roi_xmax = spixels
  roi_ymin = 0
  roi_ymax = fpixels
  maskimage = None
  detector_thickstep = detector_thickstep_mm/1000
  Odet = 0.000000
  if vectorize:
    spindle_vector = new_array([0,0,1])
    pix0_vector = new_array(pix0_vector_mm)/1000
    fdet_vector = new_array(fdet_vector)
    sdet_vector = new_array(sdet_vector)
    odet_vector = new_array(odet_vector)
    beam_vector = new_array(beam_vector)
    polar_vector = new_array(polar_vector)
  else:
    fdet_vector = convert_vector(fdet_vector, use_numpy)
    sdet_vector = convert_vector(sdet_vector, use_numpy)
    odet_vector = convert_vector(odet_vector, use_numpy)
    pix0_vector = convert_vector(pix0_vector_mm, use_numpy)/1000
    beam_vector = convert_vector(beam_vector, use_numpy)
    polar_vector = convert_vector(polar_vector, use_numpy)
    spindle_vector = new_array([0,0,0,1])
  distance = distance_mm/1000
  close_distance = close_distance_mm/1000
  detector_thick = detector_thick_mm/1000
  detector_attnlen = 0.000234
  sources = 1
  source_X = new_array([-10.000000])
  source_Y = new_array([0.000000])
  source_Z  = new_array([0.000000])
  source_I = new_array([1.000000])
  source_lambda = new_array([1e-10])
  dmin = 0.000000
  phi0 = 0.000000
  phistep = 0.000000
  mosaic_spread = 0.000000
  mosaic_umats = new_array([1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0])
  xtal_shape = 'TOPHAT'
  if vectorize:
    a0 = new_array([3.86524e-09, -2.18873e-09, 2.29551e-09])
    b0 = new_array([3.7375e-09, 3.96373e-09, -2.51395e-09])
    c0 = new_array([-8.39164e-10, 4.26918e-09, 5.4836e-09])
    ap = new_array([3.86524e-09, -2.18873e-09, 2.29551e-09])
    bp = new_array([3.7375e-09, 3.96373e-09, -2.51395e-09]) 
    cp = new_array([-8.39164e-10, 4.26918e-09, 5.4836e-09])
  else:
    a0 = new_array([5e-09, 3.86524e-09, -2.18873e-09, 2.29551e-09])
    b0 = new_array([6e-09, 3.7375e-09, 3.96373e-09, -2.51395e-09])
    c0 = new_array([7e-09, -8.39164e-10, 4.26918e-09, 5.4836e-09])
    ap = new_array([5e-09, 3.86524e-09, -2.18873e-09, 2.29551e-09])
    bp = new_array([6e-09, 3.7375e-09, 3.96373e-09, -2.51395e-09]) 
    cp = new_array([7e-09, -8.39164e-10, 4.26918e-09, 5.4836e-09])
  Na = Ncells_abc[0]
  Nb = Ncells_abc[1]
  Nc = Ncells_abc[2]
  fudge = 1
  V_cell = 2.100000e5
  Xbeam=0.00505 
  Ybeam=0.00505
  interpolate = False
  h_max= 31
  h_min= -31
  k_max= 37
  k_min= -37
  l_max= 43
  l_min= -43
  

  Fhkl_indices = [tuple(h) for h in Fhkl_indices]
  Fhkl = {h:v for h,v in zip(Fhkl_indices,Fhkl_data)}
  Fhkl = Fhkl_remove(Fhkl, h_max, h_min, k_max, k_min, l_max, l_min)
  Fhkl_mat = Fhkl_dict_to_mat(Fhkl, h_max, h_min, k_max, k_min, l_max, l_min, default_F, prefix)
  if vectorize:
    from torchBragg.forward_simulation.vectorized.diffraction_vectorized import add_torchBragg_spots
    Fhkl_input = Fhkl_mat
  else:
    from torchBragg.forward_simulation.naive.diffraction import add_torchBragg_spots
    Fhkl_input = Fhkl

  raw_pixels = add_torchBragg_spots(spixels, 
                      fpixels,
                      phisteps,
                      mosaic_domains,
                      oversample,
                      pixel_size,
                      roi_xmin, roi_xmax, roi_ymin, roi_ymax,
                      maskimage, 
                      detector_thicksteps,
                      spot_scale, fluence,
                      detector_thickstep,
                      Odet,
                      fdet_vector, sdet_vector, odet_vector, pix0_vector,
                      curved_detector, distance, beam_vector, close_distance,
                      point_pixel,
                      detector_thick, detector_attnlen,
                      sources,
                      source_X, source_Y, source_Z, source_lambda,
                      dmin,phi0, phistep,
                      a0, b0, c0, ap, bp, cp, spindle_vector,
                      mosaic_spread,
                      mosaic_umats,
                      xtal_shape,
                      Na, Nb, Nc,
                      fudge,
                      integral_form,
                      V_cell,
                      Xbeam, Ybeam,
                      interpolate,
                      h_max, h_min, k_max, k_min, l_max, l_min,
                      Fhkl_input, default_F,
                      nopolar,source_I,
                      polarization,
                      polar_vector,
                      verbose=verbose,
                      use_numpy=use_numpy,
                      device="cpu")
  raw_pixels *= 64e9

  if add_background_bool:

    if vectorize:
      from torchBragg.forward_simulation.vectorized.add_background_vectorized import add_background
    else:
      from torchBragg.forward_simulation.naive.add_background import add_background

    # add background of water
    Fmap_pixel = False
    override_source = -1
    amorphous_molecules= 33456343277777776.000000
    

    stols = 18

    stol_of = new_array([-1e+99, -1e+98, 0, 3.65e+08, 7e+08, 1.2e+09, 1.62e+09, 2e+09, 1.8e+09, 2.16e+09, 2.36e+09, 2.8e+09,
              3e+09, 3.45e+09, 4.36e+09, 5e+09, 1e+98, 1e+99])
    Fbg_of = new_array([2.57, 2.57, 2.57, 2.58, 2.8, 5, 8, 6.75, 7.32, 6.75, 6.5, 4.5, 4.3, 4.36, 3.77, 3.17, 3.17, 3.17])



    background_pixels, invalid_pixel = add_background(oversample, 
                                                      override_source,
                                                      sources,
                                                      spixels,
                                                      fpixels,
                                                      pixel_size,
                                                      roi_xmin, roi_xmax, roi_ymin, roi_ymax,
                                                      detector_thicksteps,
                                                      fluence, amorphous_molecules, 
                                                      Fmap_pixel, # bool override: just plot interpolated structure factor at every pixel, useful for making absorption masks
                                                      detector_thickstep, Odet, 
                                                      fdet_vector, sdet_vector, odet_vector, 
                                                      pix0_vector, curved_detector, distance, beam_vector,
                                                      close_distance, point_pixel, detector_thick, detector_attnlen,
                                                      source_I, source_X, source_Y, source_Z, source_lambda,
                                                      stol_of, stols, Fbg_of, nopolar, polarization, polar_vector,
                                                      verbose, use_numpy, device="cpu"
                                                      )
    raw_pixels += background_pixels


    # add background of air
    Fmap_pixel = False
    override_source = -1

    amorphous_molecules = 14051664176666668.000000
    stols = 9
    
    stol_of = new_array([-1e+99, -1e+98, 0, 4.5e+08, 1.74e+09, 3.5e+09, 5e+09, 1e+98, 1e+99])
    Fbg_of = new_array([14.1, 14.1, 14.1, 13.5, 8.35, 4.78, 4.22, 4.22, 4.22])
    

    background_pixels, invalid_pixel = add_background(oversample, 
                                                      override_source,
                                                      sources,
                                                      spixels,
                                                      fpixels,
                                                      pixel_size,
                                                      roi_xmin, roi_xmax, roi_ymin, roi_ymax,
                                                      detector_thicksteps,
                                                      fluence, amorphous_molecules, 
                                                      Fmap_pixel, # bool override: just plot interpolated structure factor at every pixel, useful for making absorption masks
                                                      detector_thickstep, Odet, 
                                                      fdet_vector, sdet_vector, odet_vector, 
                                                      pix0_vector, curved_detector, distance, beam_vector,
                                                      close_distance, point_pixel, detector_thick, detector_attnlen,
                                                      source_I, source_X, source_Y, source_Z, source_lambda,
                                                      stol_of, stols, Fbg_of, nopolar, polarization, polar_vector,
                                                      verbose, use_numpy, device="cpu"
                                                      )
    raw_pixels += background_pixels
  if noise_params is not None:
    quantum_gain, adc_offset_adu, detector_calibration_noise_pct, flicker_noise_pct, readout_noise_adu, \
    detector_psf_type, detector_psf_fwhm_mm, detector_psf_kernel_radius_pixels = noise_params
    raw_pixels = add_noise(raw_pixels, flicker_noise_pct/100., detector_calibration_noise_pct/100., 
                           prefix.randn_like(raw_pixels), readout_noise_adu,
                           quantum_gain, adc_offset_adu, detector_psf_type.name.lower(), detector_psf_fwhm_mm, pixel_size_mm, 
                           detector_psf_kernel_radius_pixels, convolution_type)
  return(raw_pixels)

if __name__=="__main__":
  spixels = 1024
  fpixels = 1024
  use_numpy = False
  vectorize = True
  add_background_bool = False
  add_noise_bool = False

  if add_noise_bool:
    use_numpy = False
    print('Must use PyTorch for add_noise')

  start_time = time.time()
  raw_pixels_0, params, noise_params = tst_nanoBragg_basic(spixels,fpixels, add_background_bool, add_noise_bool)
  end_time = time.time()
  print("nanoBragg time: ", end_time-start_time)
  raw_pixels_0 = raw_pixels_0.as_numpy_array()

  start_time = time.time()
  raw_pixels_1 = tst_torchBragg_basic(spixels,fpixels, params, use_numpy, vectorize, add_background_bool, noise_params)
  end_time = time.time()
  print("torchBragg time: ", end_time-start_time)

  if not(use_numpy):
    raw_pixels_1 = raw_pixels_1.numpy()
  
  plt.figure(); plt.imshow(raw_pixels_0, vmax=300);plt.colorbar(); plt.savefig("nanoBragg_basic.png")
  plt.figure(); plt.imshow(raw_pixels_1, vmax=300);plt.colorbar(); plt.savefig("nanoBragg_basic1.png")

  fig, axs = plt.subplots(1, 2)
  im = axs[0].imshow(raw_pixels_0,vmax=300)
  # cbar = fig.colorbar(im, ax=axs[0])

  im2 = axs[1].imshow(raw_pixels_1, vmax=300)
  # cbar2 = fig.colorbar(im2, ax=axs[1])
  plt.savefig("nanoBragg_vs_torchBragg_basic.png")

#   assert(np.mean(np.abs(raw_pixels_0-raw_pixels_1))/np.mean(raw_pixels_0) < 1e-7)

#   print("OK")
