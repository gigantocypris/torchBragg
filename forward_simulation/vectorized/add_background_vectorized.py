import torch
import numpy as np
from torchBragg.forward_simulation.naive.utils import rotate_axis, unitize, dot_product, magnitude, polint, polarization_factor, r_e_sqr, which_package
from torchBragg.forward_simulation.vectorized.diffraction_vectorized import simulation_setup
from torchBragg.forward_simulation.vectorized.utils_vectorized import unitize_vectorized, polarization_factor_vectorized

def add_background(oversample, 
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
                   verbose, use_numpy, device,
                   ):
    
    prefix, new_array = which_package(use_numpy)
    subpixel_size = pixel_size/oversample

    if override_source>=0:
        raise NotImplementedError("override_source not implemented")
    
    if Fmap_pixel:
        raise NotImplementedError("Fmap_pixel not implemented")
    
    # make sure we are normalizing with the right number of sub-steps
    steps = oversample*oversample

    diffracted_mat, capture_fraction, omega_pixel, scattering_mat, incident_mat, stol = \
    simulation_setup(prefix, spixels, fpixels, oversample, subpixel_size, detector_thicksteps, detector_thickstep,
                     fdet_vector, sdet_vector, odet_vector, pix0_vector, curved_detector, pixel_size, close_distance,
                     point_pixel, detector_thick, detector_attnlen, source_X, source_Y, source_Z, source_lambda, device)

    # now we need to find the nearest four "stol file" points
    dist = stol_of[2:-3][:,None,None,None,None]-stol[None,:,:,:,:] # add a 0th dimension to stol
    # dist is stol_of[2:-3] x spixels x fpixels x detector_thicksteps x sources
    dist[dist>0] = prefix.max(prefix.abs(dist)) + 1

    nearest = prefix.argmin(prefix.abs(dist),axis=0)+2

    stol_of_mat = prefix.stack([stol_of[nearest-1], stol_of[nearest], stol_of[nearest+1], stol_of[nearest+2]],axis=0)
    Fbg_of_mat = prefix.stack([Fbg_of[nearest-1], Fbg_of[nearest], Fbg_of[nearest+1], Fbg_of[nearest+2]],axis=0)
    # stol_of_mat is 4 x spixels x fpixels x detector_thicksteps x sources
    # Fbg_of_mat is 4 x spixels x fpixels x detector_thicksteps x sources

    # cubic spline interpolation
    Fbg = polint(stol_of_mat, Fbg_of_mat, stol)
    # Fbg is the structure factor for this pixel

    # allow negative F values to yield negative intensities
    sign = prefix.ones_like(Fbg)
    sign[Fbg<0.0] = -1.0
    
    
    # polarization factor
    if(nopolar):
        polar = 1.0
    else:
        # need to compute polarization factor
        polar = polarization_factor_vectorized(polarization,incident_mat,diffracted_mat,polar_vector, use_numpy)
    # polar is subpixels_x, subpixels_y, detector_thicksteps, sources
    
    # accumulate unscaled pixel intensity from this

    # sign is subpixels_x, subpixels_y, detector_thicksteps, sources
    # Fbg is subpixels_x, subpixels_y, detector_thicksteps, sources
    # polar is subpixels_x, subpixels_y, detector_thicksteps, sources
    # omega_pixel is subpixels_x, subpixels_y, detector_thicksteps
    # capture_fraction is subpixels_x, subpixels_y, detector_thicksteps
    # source_I is num_sources
    raw_pixels = sign*Fbg*Fbg*polar*omega_pixel[:,:,:,None]*capture_fraction[:,:,:,None]*source_I[None,None,None,:]

    raw_pixels = prefix.sum(raw_pixels, axis=(2,3)) # sum over detector_thicksteps, sources
    # raw_pixels is spixels, fpixels

    # sum over each oversampled tile
    raw_pixels = raw_pixels.reshape((spixels, oversample, fpixels, oversample))
    raw_pixels = prefix.sum(raw_pixels, axis=(1,3)) # sum over oversample_x, oversample_y

    raw_pixels = raw_pixels*r_e_sqr*fluence*amorphous_molecules/steps
    
    return(raw_pixels, 0) # invalid_pixel not calculated and returned as 0
