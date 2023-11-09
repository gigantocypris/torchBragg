from __future__ import absolute_import, division, print_function
# absolute bare-minimum diffraction image simulation

from simtbx.nanoBragg import nanoBragg
from diffraction import add_torchBragg_spots
from utils import which_package
import numpy as np
import torch
import matplotlib.pyplot as plt
from simtbx.nanoBragg import shapetype

torch.set_default_dtype(torch.float64)

def tst_nanoBragg_minimal(spixels,fpixels):
    # create the simulation object, all parameters have sensible defaults
    SIM = nanoBragg(detpixels_slowfast=(spixels,fpixels))
    # SIM.seed = 10
    # SIM.randomize_orientation()
    # dont bother with importing a structure, we just want spots
    SIM.default_F = 1
    SIM.F000 = 10

    # default is one unit cell, let's do 125
    SIM.Ncells_abc = (5,5,5)
    SIM.xtal_shape = shapetype.Tophat
    # default orientation is with a axis down the beam, lets pick a random one
    # SIM.randomize_orientation()

    # display randomly-picked missetting angles
    print(SIM.missets_deg)
    # or an Arndt-Wonacott A matrix (U*B), same as used by mosflm
    print(SIM.Amatrix) # unit cell encoded in this matrix
    # show all parameters
    SIM.show_params()
    # now actually run the simulation
    SIM.add_nanoBragg_spots()

    # # write out a file on arbitrary scale, header contains beam center in various conventions
    # SIM.to_smv_format(fileout="intimage_001.img")

    # # now apply a scale to get more than 1 photon/pixel and add noise
    # SIM.raw_pixels*=2000
    # SIM.add_noise()
    # SIM.to_smv_format(fileout="noiseimage_001.img")
    breakpoint()
    return SIM.raw_pixels

def tst_torchBragg_minimal(spixels,fpixels, use_numpy=False):
    prefix, new_array = which_package(use_numpy)

    phisteps = 1
    mosaic_domains = 1
    oversample = 2
    pixel_size = 0.0001
    roi_xmin = 0 
    roi_xmax = 1024
    roi_ymin = 0
    roi_ymax = 1024
    maskimage = None
    detector_thicksteps = 1
    spot_scale = 1
    fluence = 125932015286227086360700780544.0
    detector_thickstep = 0.000000
    Odet = 0.000000
    fdet_vector = new_array([0.,0,0,1.]) 
    sdet_vector = new_array([0.,0,-1.,0]) 
    odet_vector = new_array([0.,1.,0,0]) 
    pix0_vector = new_array([0.000000, 0.100000, 0.051300, -0.051300])
    curved_detector = False
    distance = 0.1 
    beam_vector =  new_array([0,1.,0,0]) 
    close_distance = 0.100000
    point_pixel = False
    detector_thick = 0.000000
    detector_attnlen = 0.000234
    sources = 1
    source_X = new_array([-10.000000])
    source_Y = new_array([0.000000])
    source_Z  = new_array([0.000000])
    source_lambda = new_array([1e-10])
    dmin = 0.000000
    phi0 = 0.000000
    phistep = 0.000000

    # a0 = new_array([5e-09, 3.86524e-09, -2.18873e-09, 2.29551e-09])
    # b0 = new_array([6e-09, 3.7375e-09, 3.96373e-09, -2.51395e-09])
    # c0 = new_array([7e-09, -8.39164e-10, 4.26918e-09, 5.4836e-09])
    # ap = new_array([5e-09, 3.86524e-09, -2.18873e-09, 2.29551e-09])
    # bp = new_array([6e-09, 3.7375e-09, 3.96373e-09, -2.51395e-09]) 
    # cp = new_array([7e-09, -8.39164e-10, 4.26918e-09, 5.4836e-09])

    a0 = new_array([7.8e-09, 7.8e-09, 4.77612e-25, 4.77612e-25])
    b0 = new_array([7.8e-09, 0, 7.8e-09, 4.77612e-25])
    c0 = new_array([3.8e-09, 0, 0, 3.8e-09])
    ap = new_array([7.8e-09, 7.8e-09, 4.77612e-25, 4.77612e-25])
    bp = new_array([7.8e-09, 0, 7.8e-09, 4.77612e-25]) 
    cp = new_array([3.8e-09, 0, 0, 3.8e-09])

    # a0 = np.array([7.8e-09, 7.8e-09, 4.77612e-25, 4.77612e-25])
    # b0 = np.array([7.8e-09, 0, 7.8e-09, 4.77612e-25])
    # c0 = np.array([3.8e-09, 0, 0, 3.8e-09])
    # ap = np.array([7.8e-09, 7.8e-09, 4.77612e-25, 4.77612e-25])
    # bp = np.array([7.8e-09, 0, 7.8e-09, 4.77612e-25]) 
    # cp = np.array([3.8e-09, 0, 0, 3.8e-09])

    # a0 = new_array([7.8e-09, 7.8e-09, 0, 0])
    # b0 = new_array([7.8e-09, 0, 7.8e-09, 0])
    # c0 = new_array([3.8e-09, 0, 0, 3.8e-09])
    # ap = new_array([7.8e-09, 7.8e-09, 0, 0])
    # bp = new_array([7.8e-09, 0, 7.8e-09, 0]) 
    # cp = new_array([3.8e-09, 0, 0, 3.8e-09])

    spindle_vector = new_array([0,0,0,1.])
    mosaic_spread = 0.000000
    # mosaic_umats = new_array([[1.0, 0, 0],[0, 1.0, 0],[0, 0, 1.0]])
    mosaic_umats = new_array([1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0])
    xtal_shape = 'TOPHAT' #'SQUARE'
    Na = 5.0
    Nb = 5.0
    Nc = 5.0
    fudge = 1
    integral_form = 0
    V_cell = 231192.000000
    Xbeam = 0.05125
    Ybeam = 0.05125
    interpolate = False
    h_max = 0
    h_min = 0
    k_max = 0
    k_min = 0
    l_max = 0
    l_min = 0
    Fhkl_indices = [(0,0,0)]
    Fhkl_data = [10.0]
    default_F = 1.0
    nopolar = False
    source_I = new_array([1.000000])
    polarization = 0
    polar_vector = new_array([0,0,0,1])
    verbose=9

    Fhkl = {h:v for h,v in zip(Fhkl_indices,Fhkl_data)}

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
                        Fhkl, default_F,
                        nopolar,source_I,
                        polarization,
                        polar_vector,
                        verbose=verbose,
                        use_numpy=use_numpy)
    return raw_pixels

if __name__=="__main__":
  spixels = 100 #1024    
  fpixels = 100 #1024
  use_numpy = True

  raw_pixels_0 = tst_nanoBragg_minimal(spixels, fpixels).as_numpy_array()
  raw_pixels_1 = tst_torchBragg_minimal(spixels, fpixels, use_numpy=use_numpy)

  if not(use_numpy):
    raw_pixels_1 = raw_pixels_1.numpy()

  # figure with 2 subplots
  fig, axs = plt.subplots(1, 2)
  axs[0].imshow(raw_pixels_0, vmax=0.001)
  axs[1].imshow(raw_pixels_1, vmax=0.001)
  plt.savefig("nanoBragg_vs_torchBragg.png")

  fig, axs = plt.subplots(1, 2)
  axs[0].imshow(np.log(raw_pixels_0))
  axs[1].imshow(np.log(raw_pixels_1))
  plt.savefig("nanoBragg_vs_torchBragg_log.png")

  assert(np.mean(raw_pixels_0-raw_pixels_1)/np.mean(raw_pixels_0) < 1e-9)
  print("OK")
