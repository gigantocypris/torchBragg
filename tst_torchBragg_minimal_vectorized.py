from __future__ import absolute_import, division, print_function
# absolute bare-minimum diffraction image simulation

from simtbx.nanoBragg import nanoBragg
from diffraction_vectorized import Fhkl_remove
from utils import which_package
import numpy as np
import torch
import matplotlib.pyplot as plt
from simtbx.nanoBragg import shapetype
import time

torch.set_default_dtype(torch.float64)

def tst_nanoBragg_minimal(spixels, fpixels, randomize_orientation=True, tophat=True):
    # create the simulation object, all parameters have sensible defaults
    SIM = nanoBragg(detpixels_slowfast=(spixels,fpixels),pixel_size_mm=0.1,Ncells_abc=(5,5,5),verbose=9)
    # SIM = nanoBragg()

    if randomize_orientation:
        SIM.seed = 10
        SIM.randomize_orientation()

    # don't bother with importing a structure, we just want spots
    SIM.default_F = 1
    SIM.F000 = 10

    # default is one unit cell, let's do 125
    SIM.Ncells_abc = (5,5,5)
    if tophat:
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
    return SIM.raw_pixels, SIM.pix0_vector_mm

def tst_torchBragg_minimal(spixels, fpixels, pix0_vector_mm, use_numpy=True, randomize_orientation=True, tophat=True, vectorize=True):
    prefix, new_array = which_package(use_numpy)

    phisteps = 1
    mosaic_domains = 1
    oversample = 2
    pixel_size = 0.0001
    roi_xmin = 0 
    roi_xmax = spixels
    roi_ymin = 0
    roi_ymax = fpixels
    maskimage = None
    detector_thicksteps = 1
    spot_scale = 1
    fluence = 125932015286227086360700780544.0
    detector_thickstep = 0.000000
    Odet = 0.000000
    fdet_vector = new_array([0,0,1]) 
    sdet_vector = new_array([0,-1,0]) 
    odet_vector = new_array([1,0,0]) 
    pix0_vector = new_array([pix0_vector_mm[0]/1e3, pix0_vector_mm[1]/1e3, pix0_vector_mm[2]/1e3])
    curved_detector = False
    distance = 0.1 
    beam_vector =  new_array([1,0,0]) 
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

    if randomize_orientation:
        # with randomize_orientation(), seed = 10
        a0 = new_array([6.02977e-09, -3.41442e-09, 3.581e-09])
        b0 = new_array([4.85875e-09, 5.15285e-09, -3.26813e-09])
        c0 = new_array([-4.55546e-10, 2.31756e-09, 2.97681e-09])
        ap = new_array([6.02977e-09, -3.41442e-09, 3.581e-09])
        bp = new_array([4.85875e-09, 5.15285e-09, -3.26813e-09]) 
        cp = new_array([-4.55546e-10, 2.31756e-09, 2.97681e-09])
    else:
        # without randomize_orientation()
        a0 = new_array([7.8e-09, 0, 0])
        b0 = new_array([0, 7.8e-09, 0])
        c0 = new_array([0, 0, 3.8e-09])
        ap = new_array([7.8e-09, 0, 0])
        bp = new_array([0, 7.8e-09, 0]) 
        cp = new_array([0, 0, 3.8e-09])

    spindle_vector = new_array([0,0,1])
    mosaic_spread = 0.000000
    # mosaic_umats = new_array([[1.0, 0, 0],[0, 1.0, 0],[0, 0, 1.0]])
    mosaic_umats = new_array([1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0])
    if tophat:
       xtal_shape = 'TOPHAT' 
    else:
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
    source_I = new_array([1.000000])
    polarization = 0
    polar_vector = new_array([0,0,1])
    verbose=9

    Fhkl = {h:v for h,v in zip(Fhkl_indices,Fhkl_data)}
    Fhkl = Fhkl_remove(Fhkl, h_max, h_min, k_max, k_min, l_max, l_min)

    if vectorize:
        from diffraction_vectorized import add_torchBragg_spots
    else:
        from diffraction import add_torchBragg_spots

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
                        dmin, phi0, phistep,
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
    
    use_numpy = False
    randomize_orientation = False
    tophat = False
    vectorize = True

    # does not work for modified sizes
    spixels = 128
    fpixels = 128

    raw_pixels_0, pix0_vector_mm = tst_nanoBragg_minimal(spixels,fpixels, randomize_orientation=randomize_orientation, tophat=tophat)
    raw_pixels_1 = tst_torchBragg_minimal(spixels,fpixels, pix0_vector_mm, randomize_orientation=randomize_orientation, tophat=tophat, use_numpy=use_numpy, vectorize=vectorize)
    
    raw_pixels_0 = raw_pixels_0.as_numpy_array()
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

    assert(np.mean(np.abs(raw_pixels_0-raw_pixels_1))/np.mean(raw_pixels_0) < 1e-10)
    print("OK")
