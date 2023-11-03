import numpy as np
from utils import rotate_axis, unitize, dot_product, magnitude, polint, polarization_factor, r_e_sqr

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
                   source_I, source_X, source_Y, source_Z, source_wavelength,
                   stol_of, stols, Fbg_of, nopolar, polarization, polar_vector,
                   verbose,
                   ):
    
    source_start = 0
    orig_sources = sources
    end_sources = sources
    max_I = 0.0
    raw_pixels = np.zeros((spixels,fpixels))
    invalid_pixel = np.zeros((spixels,fpixels),dtype=bool)
    have_single_source = False

    if override_source>=0:
        # user-specified idx_single_source in the argument
        source_start = override_source
        end_sources = source_start + 1
        have_single_source = True
    
    # make sure we are normalizing with the right number of sub-steps
    steps = oversample*oversample
    subpixel_size = pixel_size/oversample

    # sweep over detector
    sum = sumsqr = 0.0
    sumn = 0
    progress_pixel = 0
    omega_sum = 0.0
    nearest = 0
    i = 0

    for spixel in range(spixels):
        for fpixel in range(fpixels):
            # allow for just one part of detector to be rendered
            if fpixel < roi_xmin or fpixel > roi_xmax or spixel < roi_ymin or spixel > roi_ymax:
                invalid_pixel[spixel,fpixel] = True
                i += 1
                print("skipping pixel %d %d\n",spixel,fpixel)
            else:
                # reset background photon count for this pixel
                Ibg = 0

                # loop over sub-pixels
                for subS in range(oversample):
                    for subF in range(oversample):
                        
                        # absolute mm position on detector (relative to its origin)
                        Fdet = subpixel_size*(fpixel*oversample + subF) + subpixel_size/2.0
                        Sdet = subpixel_size*(spixel*oversample + subS) + subpixel_size/2.0

                        for thick_tic in range(detector_thicksteps):
                            Ibg_contribution, Fbg = get_thickness_contribution(thick_tic,
                                                                detector_thickstep,
                                                                Fdet, Sdet, Odet,
                                                                fdet_vector, sdet_vector, odet_vector,
                                                                pix0_vector,
                                                                curved_detector,
                                                                distance,
                                                                beam_vector,
                                                                pixel_size,
                                                                close_distance,
                                                                point_pixel,
                                                                omega_sum,
                                                                detector_thick,
                                                                detector_attnlen,
                                                                source_start, end_sources,
                                                                have_single_source, source_I,
                                                                orig_sources, source_X, source_Y, source_Z,
                                                                source_wavelength, nearest, stol_of, stols, Fbg_of,
                                                                nopolar, polarization, polar_vector, verbose, i)
                            Ibg += Ibg_contribution
                
                # save photons/pixel (if fluence specified), or F^2/omega if no fluence given
                raw_pixels[spixel, fpixel] = Ibg*r_e_sqr*fluence*amorphous_molecules/steps
                
                # override: just plot interpolated structure factor at every pixel, useful for making absorption masks
                if Fmap_pixel:
                    raw_pixels[spixel, fpixel] = Fbg
                
                # keep track of basic statistics
                if raw_pixels[spixel, fpixel] > max_I or i==0:
                    max_I = raw_pixels[spixel, fpixel]
                    max_I_x = Fdet
                    max_I_y = Sdet
                
                sum += raw_pixels[spixel, fpixel]
                sumsqr += raw_pixels[spixel, fpixel]*raw_pixels[spixel, fpixel]
                sumn += 1

                # debugging infrastructure
                # NOT IMPLEMENTED
                # end of debugging infrastructure

                i += 1

    if verbose:
        print("solid angle subtended by detector = %g steradian ( %g%% sphere)" % (omega_sum/steps,100*omega_sum/steps/4/np.pi))
        print("max_I= %g @ ( %g, %g) sum= %g avg= %g" % (max_I,max_I_x,max_I_y,sum,sum/sumn))

    return(raw_pixels, invalid_pixel)

def get_thickness_contribution(thick_tic,
                                detector_thickstep,
                                Fdet, Sdet, Odet,
                                fdet_vector, sdet_vector, odet_vector,
                                pix0_vector,
                                curved_detector,
                                distance,
                                beam_vector,
                                pixel_size,
                                close_distance,
                                point_pixel,
                                omega_sum,
                                detector_thick,
                                detector_attnlen,
                                source_start, end_sources,
                                have_single_source, source_I,
                                orig_sources, source_X, source_Y, source_Z,
                                source_wavelength, nearest, stol_of, stols, Fbg_of,
                                nopolar, polarization, polar_vector, verbose, i
                                ):
    
    Ibg = 0
    # assume "distance" is to the front of the detector sensor layer
    Odet = thick_tic*detector_thickstep
    pixel_pos = np.zeros([4,])

    # construct detector pixel position in 3D space
    pixel_pos[1] = Fdet*fdet_vector[1]+Sdet*sdet_vector[1]+Odet*odet_vector[1]+pix0_vector[1]
    pixel_pos[2] = Fdet*fdet_vector[2]+Sdet*sdet_vector[2]+Odet*odet_vector[2]+pix0_vector[2]
    pixel_pos[3] = Fdet*fdet_vector[3]+Sdet*sdet_vector[3]+Odet*odet_vector[3]+pix0_vector[3]

    if curved_detector:
        # construct detector pixel that is always "distance" from the sample
        vector = np.zeros([4,])
        vector[1] = distance*beam_vector[1]
        vector[2] = distance*beam_vector[2]
        vector[3] = distance*beam_vector[3]

        # treat detector pixel coordinates as radians
        newvector = rotate_axis(vector,sdet_vector,pixel_pos[2]/distance)
        pixel_pos = rotate_axis(newvector,fdet_vector,pixel_pos[3]/distance)

    # construct the diffracted-beam unit vector to this pixel
    airpath, diffracted = unitize(pixel_pos)
    
    # solid angle subtended by a pixel: (pix/airpath)^2*cos(2theta)
    omega_pixel = pixel_size*pixel_size/airpath/airpath*close_distance/airpath

    # option to turn off obliquity effect, inverse-square-law only
    if point_pixel:
        omega_pixel = 1.0/airpath/airpath
    omega_sum += omega_pixel

    if detector_thick > 0.0:
        # inverse of effective thickness increase
        parallax = dot_product(diffracted,odet_vector)
        capture_fraction = np.exp(-thick_tic*detector_thickstep/detector_attnlen/parallax) - np.exp(-(thick_tic+1)*detector_thickstep/detector_attnlen/parallax)
    else:
        capture_fraction = 1.0

    # loop over sources now
    for source in range(source_start, end_sources):
        Ibg_contribution, Fbg = get_source_contribution(source,
                            have_single_source,
                            orig_sources,
                            source_I,
                            source_X, source_Y, source_Z,
                            source_wavelength, diffracted,
                            nearest, stol_of, stols, Fbg_of,
                            nopolar, polarization, polar_vector,
                            omega_pixel, capture_fraction, verbose, i,
                            )
        Ibg += Ibg_contribution
    return(Ibg, Fbg)

def get_source_contribution(source,
                            have_single_source,
                            orig_sources,
                            source_I,
                            source_X, source_Y, source_Z,
                            source_wavelength, diffracted,
                            nearest, stol_of, stols, Fbg_of,
                            nopolar, polarization, polar_vector,
                            omega_pixel, capture_fraction, verbose, i,
                           ):    
    if have_single_source:
        n_source_scale = orig_sources
    else:
        n_source_scale = source_I[source]

    incident = np.zeros([4,])
    incident[1] = -source_X[source]
    incident[2] = -source_Y[source]
    incident[3] = -source_Z[source]

    # lambda --> wavelength, C++ code --> Python code
    wavelength = source_wavelength[source]

    # construct the incident beam unit vector while recovering source distance
    source_path, incident = unitize(incident)

    # construct the scattering vector for this pixel
    scattering = np.zeros([4,])
    scattering[1] = (diffracted[1]-incident[1])/wavelength
    scattering[2] = (diffracted[2]-incident[2])/wavelength
    scattering[3] = (diffracted[3]-incident[3])/wavelength

    # sin(theta)/lambda is half the scattering vector length
    stol = 0.5*magnitude(scattering)

    # now we need to find the nearest four "stol file" points

    dist = stol_of[2:-3]-stol
    nearest = np.argmin(np.abs(dist[dist<0]))+2

    # cubic spline interpolation
    Fbg = polint(stol_of[nearest-1:nearest+3], Fbg_of[nearest-1:nearest+3], stol)

    # allow negative F values to yield negative intensities
    sign=1.0
    if Fbg<0.0:
        sign=-1.0
    
    # Fbg is the structure factor for this pixel

    # polarization factor
    if(not(nopolar)):
        # need to compute polarization factor
        polar = polarization_factor(polarization,incident,diffracted,polar_vector)
    else:
        polar = 1.0
    
    # accumulate unscaled pixel intensity from this
    Ibg = sign*Fbg*Fbg*polar*omega_pixel*capture_fraction*n_source_scale
    if verbose>7 and i==1:
        print("DEBUG: Fbg= %g polar= %g omega_pixel= %g source[%d]= %g capture_fraction= %g" % \
              (Fbg,polar,omega_pixel,source,source_I[source],capture_fraction))    
    return(Ibg, Fbg)

