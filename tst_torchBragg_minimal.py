from __future__ import absolute_import, division, print_function
# absolute bare-minimum diffraction image simulation

from simtbx.nanoBragg import nanoBragg
from main import add_torchBragg_spots
import numpy as np
import matplotlib.pyplot as plt

def tst_nanoBragg_minimal():
    # create the simulation object, all parameters have sensible defaults
    SIM = nanoBragg()

    # dont bother with importing a structure, we just want spots
    SIM.default_F = 1
    SIM.F000 = 10

    # default is one unit cell, lets to 125
    SIM.Ncells_abc = (5,5,5)

    # default orientation is with a axis down the beam, lets pick a random one
    # SIM.randomize_orientation()

    # display randomly-picked missetting angles
    print(SIM.missets_deg)
    # or an Arndt-Wonacott A matrix (U*B), same as used by mosflm
    print(SIM.Amatrix)

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

    return SIM.raw_pixels

def tst_torchBragg_minimal():
    spixels = 1024
    fpixels = 1024
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
    r_e_sqr = 7.94079248018965e-30 # Thomson cross section in m^2
    detector_thickstep = 0.000000
    Odet = 0.000000
    fdet_vector = np.array([0,0,0,1]) 
    sdet_vector = np.array([0,0,-1,0]) 
    odet_vector = np.array([0,1,0,0]) 
    pix0_vector = np.array([0.000000, 0.100000, 0.051300, -0.051300])
    curved_detector = False
    distance = 0.1 
    beam_vector =  np.array([0,1,0,0]) 
    close_distance = 0.100000
    point_pixel = False
    detector_thick = 0.000000
    detector_attnlen = 0.000234
    sources = 1
    source_X = np.array([-10.000000])
    source_Y = np.array([0.000000])
    source_Z  = np.array([0.000000])
    source_lambda = np.array([1e-10])
    dmin = 0.000000
    phi0 = 0.000000
    phistep = 0.000000
    a0 = np.array([7.800000e-09, 6.484601e-09, 3.443972e-09, -2.632301e-09])
    b0 = np.array([7.800000e-09, -2.431495e-09, 6.811206e-09, 2.921523e-09])
    c0 = np.array([3.800000e-09, 1.748274e-09, -7.835150e-10, 3.281713e-09])
    ap = np.array([7.800000e-09, 6.484601e-09, 3.443972e-09, -2.632301e-09])
    bp = np.array([7.800000e-09, -2.431495e-09, 6.811206e-09, 2.921523e-09]) 
    cp = np.array([3.800000e-09, 1.748274e-09, -7.835150e-10, 3.281713e-09])
    spindle_vector = np.array([0,0,0,1])
    mosaic_spread = 0.000000
    mosaic_umats = np.array([[1.0, 0, 0],[0, 1.0, 0],[0, 0, 1.0]])
    mosaic_umats = np.array([1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0])
    xtal_shape = 'SQUARE'
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
    source_I = np.array([1.000000])
    polarization = 0
    polar_vector = np.array([0,0,0,1])
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
                        r_e_sqr,
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
                        verbose=verbose)
    return raw_pixels

if __name__=="__main__":
  raw_pixels_0 = tst_nanoBragg_minimal()

  breakpoint()
  
  raw_pixels_1 = tst_torchBragg_minimal()

  # figure with 2 subplots
  fig, axs = plt.subplots(1, 2)
  axs[0].imshow(raw_pixels_0.as_numpy_array(), vmax=0.001)
  axs[1].imshow(raw_pixels_1, vmax=0.001)
  plt.savefig("nanoBragg_vs_torchBragg.png")

  fig, axs = plt.subplots(1, 2)
  axs[0].imshow(np.log(raw_pixels_0.as_numpy_array()))
  axs[1].imshow(np.log(raw_pixels_1))
  plt.savefig("nanoBragg_vs_torchBragg_log.png")

  breakpoint()
  print("OK")
