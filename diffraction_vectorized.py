import numpy as np
import torch
from utils import r_e_sqr, which_package

from utils_vectorized import sincg_vectorized, sinc3_vectorized, unitize_vectorized, polarization_factor_vectorized



def add_torchBragg_spots(spixels, fpixels, phisteps, mosaic_domains, oversample, pixel_size,
                         roi_xmin, roi_xmax, roi_ymin, roi_ymax, maskimage, detector_thicksteps,
                         spot_scale, fluence, detector_thickstep, Odet,
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
                        verbose=9,
                        use_numpy=True):
    
    prefix, new_array = which_package(use_numpy)

    subpixel_size = pixel_size/oversample

    diffracted_mat, capture_fraction, omega_pixel, scattering_mat, incident_mat, stol = \
    simulation_setup(prefix, spixels, fpixels, oversample, subpixel_size, detector_thicksteps, detector_thickstep,
                     fdet_vector, sdet_vector, odet_vector, pix0_vector, curved_detector, pixel_size, close_distance,
                     point_pixel, detector_thick, detector_attnlen, source_X, source_Y, source_Z, source_lambda)


    mos_tic_vec = prefix.arange(mosaic_domains)

    # usually apply mosaic rotation after phi rotation, assume no phi rotation here because SFX

    # reshape mosaic_umats to [mosaic_domains, 3, 3]
    mosaic_umats_reshape = mosaic_umats.reshape((mosaic_domains, 3, 3))

    
    a = mosaic_umats_reshape @ ap # mosaic domains x 3
    b = mosaic_umats_reshape @ bp # mosaic domains x 3
    c = mosaic_umats_reshape @ cp # mosaic domains x 3

    # construct fractional Miller indicies
    # scattering_mat --> scattering_mat[:,:,:,:,None,:] to add the mosaic domains dimension
    h = prefix.sum(a[None,None,None,None,:,:]*scattering_mat[:,:,:,:,None,:], axis=-1)
    k = prefix.sum(b[None,None,None,None,:,:]*scattering_mat[:,:,:,:,None,:], axis=-1)
    l = prefix.sum(c[None,None,None,None,:,:]*scattering_mat[:,:,:,:,None,:], axis=-1)
    # h,k,l are subpixels_x, subpixels_y, detector_thicksteps, sources, mosaic_domains

    # round off to nearest whole index
    h0 = prefix.ceil(h-0.5)
    k0 = prefix.ceil(k-0.5)
    l0 = prefix.ceil(l-0.5)

    # structure factor of the lattice
    
    if(xtal_shape == 'SQUARE'):
        # xtal is a paralelpiped
        F_latt = prefix.ones_like(h)
        if(Na>1):
            F_latt *= sincg_vectorized(np.pi*h, Na, prefix)
        if(Nb>1):
            F_latt *= sincg_vectorized(np.pi*k, Nb, prefix)
        if(Nc>1):
            F_latt *= sincg_vectorized(np.pi*l, Nc, prefix)
    else:
        #handy radius in reciprocal space, squared
        hrad_sqr = (h-h0)*(h-h0)*Na*Na + (k-k0)*(k-k0)*Nb*Nb + (l-l0)*(l-l0)*Nc*Nc
        
        if(xtal_shape == 'ROUND'):
            # use sinc3 for elliptical xtal shape,
            # correcting for sqrt of volume ratio between cube and sphere
            F_latt = Na*Nb*Nc*0.723601254558268*sinc3_vectorized(np.pi*prefix.sqrt(hrad_sqr * fudge), prefix)
        elif(xtal_shape == 'GAUSS'):
            # fudge the radius so that volume and FWHM are similar to square_xtal spots
            F_latt = Na*Nb*Nc*prefix.exp(-( hrad_sqr / 0.63 * fudge ))
        elif (xtal_shape == 'GAUSS_ARGCHK'):
            # fudge the radius so that volume and FWHM are similar to square_xtal spots
            my_arg = hrad_sqr / 0.63 * fudge # pre-calculate to check for no Bragg signal
            F_latt = prefix.zeros_like(my_arg)
            F_latt[my_arg<35.] = Na * Nb * Nc * prefix.exp(-(my_arg[my_arg<35.]))
            F_latt[my_arg>=35.] = 0. # not expected to give performance gain on optimized C++, only on GPU
        elif(xtal_shape == 'TOPHAT'):
            # make a flat-top spot of same height and volume as square_xtal spots
            F_latt = Na*Nb*Nc*(hrad_sqr*fudge < 0.3969 )
        else:
            F_latt = prefix.ones_like(h)

    
    # Experimental: find nearest point on Ewald sphere surface?
    if(integral_form):
        raise NotImplementedError("Integral form not implemented")

    # structure factor of the unit cell
    if(interpolate):
        raise NotImplementedError("Interpolation of structure factors not implemented")
        # F_cell = interpolate_unit_cell()
    else:
        # stack for 1 line conversion to int
        stacked_hkl = prefix.stack([h0-h_min,k0-k_min,l0-l_min], axis=0)
        if use_numpy:
            stacked_hkl = stacked_hkl.astype(int)
        else:
            stacked_hkl = stacked_hkl.type(torch.int32)

        try:
            # just take nearest-neighbor
            F_cell = Fhkl[stacked_hkl[0], stacked_hkl[1], stacked_hkl[2]]
        except IndexError:
            print("max h:", h0.max(), "max k:", k0.max(), "max l:", l0.max())
            print("min h:", h0.min(), "min k:", k0.min(), "min l:", l0.min())
            raise IndexError("hkl index not found in Fhkl, change hkl ranges when constructing Fhkl matrix.")
            
    # polarization factor
    if(nopolar):
        polar = 1.0
    else:
        # need to compute polarization factor
        polar = polarization_factor_vectorized(polarization,incident_mat,diffracted_mat,polar_vector, use_numpy)
    # polar is subpixels_x, subpixels_y, detector_thicksteps, sources

    # omega_pixel is subpixels_x, subpixels_y, detector_thicksteps
    # capture_fraction is subpixels_x, subpixels_y, detector_thicksteps
    # source_I is num_sources
    # F_cell is subpixels_x, subpixels_y, detector_thicksteps, sources, mosaic_domains
    # F_latt is subpixels_x, subpixels_y, detector_thicksteps, sources, mosaic_domains

    # convert amplitudes into intensity (photons per steradian)
    # raw_subpixels is subpixels_x, subpixels_y, detector_thicksteps, sources, mosaic_domains
    raw_subpixels = F_cell*F_cell*F_latt*F_latt*source_I[None,None,None,:,None]*capture_fraction[:,:,:,None,None]*omega_pixel[:,:,:,None,None]*polar[:,:,:,:,None]


    # make sure we are normalizing with the right number of sub-steps
    steps = mosaic_domains*oversample*oversample

    # raw_pixels is spixels, fpixels
    raw_pixels = prefix.sum(raw_subpixels, axis=(2,3,4)) # sum over detector_thicksteps, sources, mosaic_domains
    # sum over each oversampled tile
    raw_pixels = raw_pixels.reshape((spixels, oversample, fpixels, oversample))
    raw_pixels = prefix.sum(raw_pixels, axis=(1,3)) # sum over subpixels_x, subpixels_y


    # make sure we are normalizing with the right number of sub-steps
    steps = mosaic_domains*oversample*oversample
    raw_pixels = raw_pixels*r_e_sqr*fluence*spot_scale/steps

    return(raw_pixels)

def simulation_setup(prefix, spixels, fpixels, oversample, subpixel_size, detector_thicksteps, detector_thickstep,
                     fdet_vector, sdet_vector, odet_vector, pix0_vector, curved_detector, pixel_size, close_distance,
                     point_pixel, detector_thick, detector_attnlen, source_X, source_Y, source_Z, source_lambda):
   # Get Fdet and Sdet (detector coordinates) for every subpixel
    s_vec = prefix.arange(spixels*oversample)
    f_vec = prefix.arange(fpixels*oversample)
    s_mat, f_mat = prefix.meshgrid(s_vec, f_vec, indexing='ij')

    Fdet_mat = subpixel_size*f_mat + subpixel_size/2.0 # function of index 0 and 1
    Sdet_mat = subpixel_size*s_mat + subpixel_size/2.0 # function of index 0 and 1
    breakpoint()
    # assume "distance" is to the front of the detector sensor layer
    Odet_vec = prefix.arange(detector_thicksteps)*detector_thickstep # function of index 2

    
    # construct detector subpixel position in 3D space
    # pixel_pos_mat is [Fdet_mat.shape[0], Fdet_mat.shape[1], len(Odet_vec), 3]
    # pixel_pos_mat is subpixels_x, subpixels_y, detector_thicksteps, 3
    pixel_pos_mat = Fdet_mat[:,:,None,None]*fdet_vector[None,None,None,:]+Sdet_mat[:,:,None,None]*sdet_vector[None,None,None,:]+Odet_vec[None,None,:,None]*odet_vector[None,None,None,:]+pix0_vector[None,None,None,:] 


    if curved_detector:
        raise NotImplementedError

    # construct the diffracted-beam unit vector to this sub-pixel
    airpath_mat, diffracted_mat = unitize_vectorized(pixel_pos_mat, prefix)


    # solid angle subtended by a pixel: (pix/airpath)^2*cos(2theta)
    # omega_pixel is subpixels_x, subpixels_y, detector_thicksteps
    omega_pixel = (pixel_size/airpath_mat)**2*close_distance/airpath_mat
    # option to turn off obliquity effect, inverse-square-law only
    if(point_pixel):
        omega_pixel = 1.0/airpath_mat/airpath_mat   

    # now calculate detector thickness effects
    if(detector_thick > 0.0 and detector_attnlen > 0.0):
        thick_tic_vec = prefix.arange(detector_thicksteps)[None,None,:]
        # inverse of effective thickness increase
        parallax_mat = prefix.sum(diffracted_mat*odet_vector[None,None,None,:], axis=-1)

        capture_fraction = prefix.exp(-thick_tic_vec*detector_thickstep/detector_attnlen/parallax_mat) - prefix.exp(-(thick_tic_vec+1)*detector_thickstep/detector_attnlen/parallax_mat)
    else:
        capture_fraction = prefix.ones_like(omega_pixel)
    # capture_fraction is subpixels_x, subpixels_y, detector_thicksteps


    incident_mat = prefix.stack([-source_X, -source_Y, -source_Z], axis=-1)

    # construct the incident beam unit vector while recovering source distance
    source_path_vec, incident_mat = unitize_vectorized(incident_mat, prefix)

    # construct the scattering vector for each pixel
    # Add sources dimension to diffracted_mat --> diffracted_mat[:,:,:,None,:]

    scattering_mat = (diffracted_mat[:,:,:,None,:] - incident_mat[None,None,None,:,:])/source_lambda[None,None,None,:,None]
    # scattering_mat is subpixels_x, subpixels_y, detector_thicksteps, sources, 3

    # sin(theta)/lambda_0 is half the scattering vector length
    stol = 0.5*prefix.sqrt(prefix.sum(scattering_mat**2,axis=-1))
    
    return(diffracted_mat, capture_fraction, omega_pixel, scattering_mat, incident_mat, stol)
