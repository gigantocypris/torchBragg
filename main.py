import torch
import numpy as np
from utils import rotate_axis, rotate_umat, dot_product, sincg, sinc3, cross_product, vector_scale, magnitude, unitize


def detector_position(subpixel_size, oversample, fpixel, spixel, subF, subS):
    """absolute mm position on detector (relative to its origin)"""
    Fdet = subpixel_size*(fpixel*oversample + subF ) + subpixel_size/2.0
    Sdet = subpixel_size*(spixel*oversample + subS ) + subpixel_size/2.0
    return Fdet, Sdet

def find_pixel_pos(Fdet, Sdet, Odet, fdet_vector, sdet_vector, odet_vector, pix0_vector,
                   curved_detector, distance, beam_vector):
    """ construct detector subpixel position in 3D space """
    pixel_pos = np.zeros(4)
    pixel_pos[1] = Fdet*fdet_vector[1]+Sdet*sdet_vector[1]+Odet*odet_vector[1]+pix0_vector[1]
    pixel_pos[2] = Fdet*fdet_vector[2]+Sdet*sdet_vector[2]+Odet*odet_vector[2]+pix0_vector[2]
    pixel_pos[3] = Fdet*fdet_vector[3]+Sdet*sdet_vector[3]+Odet*odet_vector[3]+pix0_vector[3]
    pixel_pos[0] = 0.0
    if curved_detector:
        # construct detector pixel that is always "distance" from the sample
        vector = np.zeros(4)
        vector[1] = distance*beam_vector[1]
        vector[2] = distance*beam_vector[2]
        vector[3] = distance*beam_vector[3]
        
        # treat detector pixel coordinates as radians
        newvector = rotate_axis(vector,sdet_vector,pixel_pos[2]/distance)
        pixel_pos = rotate_axis(newvector,fdet_vector,pixel_pos[3]/distance)
    return pixel_pos  

def add_nanoBragg_spots(spixels, 
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
                        verbose=9):
    max_I = 0.0
    raw_pixels = torch.zeros((spixels,fpixels))

    # make sure we are normalizing with the right number of sub-steps
    steps = phisteps*mosaic_domains*oversample*oversample
    subpixel_size = pixel_size/oversample

    sum = 0.0
    sumsqr = 0.0
    sumn = 0
    progress_pixel = 0
    omega_sum = 0.0

    pixel_linear_ind = -1
    for spixel in range(spixels):
        for fpixel in range(fpixels):

            pixel_linear_ind += 1
            # reset photon count for this pixel
            I = 0

            # allow for just one part of detector to be rendered
            if(fpixel < roi_xmin or fpixel > roi_xmax or spixel < roi_ymin or spixel > roi_ymax):
                # out-of-bounds, move on to next pixel
                print("skipping pixel %d %d\n",spixel,fpixel)
            # allow for the use of a mask
            elif(maskimage != None):
                # skip any flagged pixels in the mask
                if(maskimage[pixel_linear_ind] == 0):
                    print("skipping pixel %d %d\n",spixel,fpixel)
            else: # render pixel
                # loop over sub-pixels
                for subS in range(oversample):
                    for subF in range(oversample):
                        Fdet, Sdet = detector_position(subpixel_size, oversample, fpixel, spixel, subF, subS)
                        # loop over detector thickness
                        for thick_tic in range(detector_thicksteps):
                            I_contribution, polar = find_detector_thickstep_contribution(thick_tic)
                            I += I_contribution*polar
                # end of sub-pixel loop

                raw_pixels[spixel,fpixel] += r_e_sqr*fluence*spot_scale*I/steps
                
                if(raw_pixels[spixel,fpixel] > max_I):
                    max_I = raw_pixels[spixel,fpixel]
                    max_I_x = Fdet
                    max_I_y = Sdet
                
                sum += raw_pixels[spixel,fpixel]
                sumsqr += raw_pixels[spixel,fpixel]*raw_pixels[spixel,fpixel]
                sumn += 1

                print_pixel_output()

    
    if(verbose):
        print("done with pixel loop\n")
        print("solid angle subtended by detector = %g steradian ( %g%% sphere)\n",omega_sum/steps,100*omega_sum/steps/4/np.pi)
        print("max_I= %g sum= %g avg= %g\n",max_I,sum,sum/sumn)


  
    
def find_detector_thickstep_contribution(thick_tic,
                                         detector_thickstep,
                                         Fdet, Sdet, Odet,
                                         fdet_vector, sdet_vector, odet_vector, pix0_vector,
                                         curved_detector, distance,
                                         beam_vector, pixel_size, close_distance,
                                         point_pixel,
                                         detector_thick, detector_attnlen,
                                         sources,
                                         ):
    # intensity for this detector thickness step, in the subpixel
    I_contribution = 0

    # assume "distance" is to the front of the detector sensor layer
    Odet = thick_tic*detector_thickstep

    # construct detector subpixel position in 3D space
    pixel_pos = find_pixel_pos(Fdet, Sdet, Odet, fdet_vector, sdet_vector, odet_vector, pix0_vector,
                   curved_detector, distance, beam_vector)

    # construct the diffracted-beam unit vector to this sub-pixel
    airpath, diffracted = unitize(pixel_pos)

    # solid angle subtended by a pixel: (pix/airpath)^2*cos(2theta)
    omega_pixel = pixel_size*pixel_size/airpath/airpath*close_distance/airpath
    # option to turn off obliquity effect, inverse-square-law only
    if(point_pixel):
        omega_pixel = 1.0/airpath/airpath
        omega_sum += omega_pixel

    # now calculate detector thickness effects
    if(detector_thick > 0.0 and detector_attnlen > 0.0):
        # inverse of effective thickness increase
        parallax = dot_product(diffracted,odet_vector)
        capture_fraction = np.exp(-thick_tic*detector_thickstep/detector_attnlen/parallax) \
                            - np.exp(-(thick_tic+1)*detector_thickstep/detector_attnlen/parallax)
    else:
        capture_fraction = 1.0


    # loop over sources now
    for source in range(sources):
        I_contribution_source = find_source_contribution(source)
        I_contribution += I_contribution_source


def find_source_contribution(source, source_X, source_Y, source_Z, source_lambda, diffracted, dmin, phisteps):
    incident = np.zeros(4)
    # retrieve stuff from cache
    incident[1] = -source_X[source]
    incident[2] = -source_Y[source]
    incident[3] = -source_Z[source]
    lambda_0 = source_lambda[source]

    # construct the incident beam unit vector while recovering source distance
    source_path, incident = unitize(incident)

    # construct the scattering vector for this pixel
    scattering = np.zeros(4)
    scattering[1] = (diffracted[1]-incident[1])/lambda_0
    scattering[2] = (diffracted[2]-incident[2])/lambda_0
    scattering[3] = (diffracted[3]-incident[3])/lambda_0

    # sin(theta)/lambda_0 is half the scattering vector length
    stol = 0.5*magnitude(scattering)

    # rough cut to speed things up when we aren't using whole detector
    if(dmin > 0.0 and stol > 0.0 and dmin > 0.5/stol):
        pass
    else: # sweep over phi angles
        for phi_tic in range(phisteps):
            find_phi_contribution()


def find_phi_contribution(phi0, phistep, phi_tic, 
                          a0, b0, c0, ap, bp, cp, spindle_vector,
                          mosaic_domains,
                          ): 
    """only 1 angle to loop over in XFEL"""
    phi = phi0 + phistep*phi_tic

    if( phi != 0.0 ):
        # rotate about spindle if neccesary
        rotate_axis(a0,ap,spindle_vector,phi)
        rotate_axis(b0,bp,spindle_vector,phi)
        rotate_axis(c0,cp,spindle_vector,phi)


    # enumerate mosaic domains
    for mos_tic in range(mosaic_domains):
        find_mosaic_domain_contribution()



def find_mosaic_domain_contribution(mosaic_spread,
                                    mosaic_umats,
                                    mos_tic,
                                    ap, bp, cp,
                                    scattering,
                                    xtal_shape,
                                    Na, Nb, Nc,
                                    fudge,
                                    integral_form,
                                    phi,
                                    V_cell,
                                    incident, 
                                    lambda_0,
                                    fdet_vector,
                                    sdet_vector,
                                    odet_vector,
                                    distance,
                                    Xbeam, Ybeam,
                                    verbose,
                                    Fdet, Sdet,
                                    interpolate,
                                    h_max, h_min, k_max, k_min, l_max, l_min,
                                    Fhkl, default_F,
                                    nopolar,
                                    ):
    # apply mosaic rotation after phi rotation
    if(mosaic_spread > 0.0):
        a = rotate_umat(ap,mosaic_umats[mos_tic*9])
        b = rotate_umat(bp,mosaic_umats[mos_tic*9])
        c = rotate_umat(cp,mosaic_umats[mos_tic*9])
    else:
        a[1]=ap[1];a[2]=ap[2];a[3]=ap[3]
        b[1]=bp[1];b[2]=bp[2];b[3]=bp[3]
        c[1]=cp[1];c[2]=cp[2];c[3]=cp[3]
    
    print(f"%d %f %f %f\n",mos_tic,mosaic_umats[mos_tic*9+0],mosaic_umats[mos_tic*9+1],mosaic_umats[mos_tic*9+2])
    print(f"%d %f %f %f\n",mos_tic,mosaic_umats[mos_tic*9+3],mosaic_umats[mos_tic*9+4],mosaic_umats[mos_tic*9+5])
    print(f"%d %f %f %f\n",mos_tic,mosaic_umats[mos_tic*9+6],mosaic_umats[mos_tic*9+7],mosaic_umats[mos_tic*9+8])

    # construct fractional Miller indicies
    h = dot_product(a,scattering)
    k = dot_product(b,scattering)
    l = dot_product(c,scattering)

    # round off to nearest whole index
    h0 = np.ceil(h-0.5)
    k0 = np.ceil(k-0.5)
    l0 = np.ceil(l-0.5)

    """
    # structure factor of the lattice (paralelpiped crystal)
    F_latt = sin(M_PI*Na*h)*sin(M_PI*Nb*k)*sin(M_PI*Nc*l)/sin(M_PI*h)/sin(M_PI*k)/sin(M_PI*l)
    """

    # structure factor of the lattice
    F_latt = 1.0
    if(xtal_shape == 'SQUARE'):
        # xtal is a paralelpiped
        if(Na>1):
            F_latt *= sincg(np.pi*h,Na)
        if(Nb>1):
            F_latt *= sincg(np.pi*k,Nb)
        if(Nc>1):
            F_latt *= sincg(np.pi*l,Nc)
    else:
        #handy radius in reciprocal space, squared
        hrad_sqr = (h-h0)*(h-h0)*Na*Na + (k-k0)*(k-k0)*Nb*Nb + (l-l0)*(l-l0)*Nc*Nc
        
    
    if(xtal_shape == 'ROUND'):
        # use sinc3 for elliptical xtal shape,
        # correcting for sqrt of volume ratio between cube and sphere
        F_latt = Na*Nb*Nc*0.723601254558268*sinc3(np.pi*np.sqrt( hrad_sqr * fudge ) )
    if(xtal_shape == 'GAUSS'):
        # fudge the radius so that volume and FWHM are similar to square_xtal spots
        F_latt = Na*Nb*Nc*np.exp(-( hrad_sqr / 0.63 * fudge ))
    if (xtal_shape == 'GAUSS_ARGCHK'):
        # fudge the radius so that volume and FWHM are similar to square_xtal spots
        my_arg = hrad_sqr / 0.63 * fudge # pre-calculate to check for no Bragg signal
        if (my_arg<35.):
            F_latt = Na * Nb * Nc * np.exp(-(my_arg))
        else:
            F_latt = 0. # not expected to give performance gain on optimized C++, only on GPU
    if(xtal_shape == 'TOPHAT'):
        # make a flat-top spot of same height and volume as square_xtal spots
        F_latt = Na*Nb*Nc*(hrad_sqr*fudge < 0.3969 );


    # no need to go further if result will be zero
    if(F_latt == 0.0):
        return F_latt


    # find nearest point on Ewald sphere surface?
    if(integral_form):
        if(phi != 0.0 or mos_tic > 0):
            # need to re-calculate reciprocal matrix

            # various cross products
            a_cross_b = cross_product(a,b)
            b_cross_c = cross_product(b,c)
            c_cross_a = cross_product(c,a)

            # new reciprocal-space cell vectors
            a_star = vector_scale(b_cross_c,1e20/V_cell)
            b_star = vector_scale(c_cross_a,1e20/V_cell)
            c_star = vector_scale(a_cross_b,1e20/V_cell)

        # reciprocal-space coordinates of nearest relp
        relp = np.zeros(4)
        relp[1] = h0*a_star[1] + k0*b_star[1] + l0*c_star[1]
        relp[2] = h0*a_star[2] + k0*b_star[2] + l0*c_star[2]
        relp[3] = h0*a_star[3] + k0*b_star[3] + l0*c_star[3]
        # d_star = magnitude(relp)

        # reciprocal-space coordinates of center of Ewald sphere
        Ewald0 = np.zeros(4)
        Ewald0[1] = -incident[1]/lambda_0/1e10
        Ewald0[2] = -incident[2]/lambda_0/1e10
        Ewald0[3] = -incident[3]/lambda_0/1e10
        # 1/lambda = magnitude(Ewald0)

        # distance from Ewald sphere in lambda=1 units
        vector = np.zeros(4)
        vector[1] = relp[1]-Ewald0[1]
        vector[2] = relp[2]-Ewald0[2]
        vector[3] = relp[3]-Ewald0[3]
        d_r = magnitude(vector)-1.0

        # unit vector of diffracted ray through relp
        _, diffracted0 = unitize(vector)

        # intersection with detector plane
        xd = dot_product(fdet_vector,diffracted0)
        yd = dot_product(sdet_vector,diffracted0)
        zd = dot_product(odet_vector,diffracted0)

        # where does the central direct-beam hit
        xd0 = dot_product(fdet_vector,incident)
        yd0 = dot_product(sdet_vector,incident)
        zd0 = dot_product(odet_vector,incident)

        # convert to mm coordinates
        Fdet0 = distance*(xd/zd) + Xbeam
        Sdet0 = distance*(yd/zd) + Ybeam

        if(verbose>8):
            print(f"integral_form: %g %g   %g %g\n",Fdet,Sdet,Fdet0,Sdet0)
        test = np.exp(-( (Fdet-Fdet0)*(Fdet-Fdet0)+(Sdet-Sdet0)*(Sdet-Sdet0) + d_r*d_r )/1e-8)
    # end of integral form


    # structure factor of the unit cell
    if(interpolate):
        # NOT IMPLEMENTED
        print("INTERPOLATE NOT IMPLEMENTED")
        # F_cell = interpolate_unit_cell()
        sys.exit()
    else:
        if ((h0<=h_max) and (h0>=h_min) and (k0<=k_max) and (k0>=k_min) and (l0<=l_max) and (l0>=l_min)):
            # just take nearest-neighbor
            F_cell = Fhkl[h0-h_min][k0-k_min][l0-l_min]
        
        else:
            F_cell = default_F # usually zero

    # now we have the structure factor for this pixel

    # polarization factor
    if(not(nopolar)):
        # need to compute polarization factor
        raise NotImplementedError("Polarization factor not implemented")
        # polar = polarization_factor(polarization,incident,diffracted,polar_vector)
    else:
        polar = 1.0

    # convert amplitudes into intensity (photons per steradian)
    I += F_cell*F_cell*F_latt*F_latt*source_I[source]*capture_fraction*omega_pixel
    return I, polar

def print_pixel_output():
    if printout:
        if((fpixel==printout_fpixel and spixel==printout_spixel) or printout_fpixel < 0):
            twotheta = atan2(sqrt(pixel_pos[2]*pixel_pos[2]+pixel_pos[3]*pixel_pos[3]),pixel_pos[1])
            test = sin(twotheta/2.0)/(lambda0*1e10)
            print(f"%4d %4d : stol = %g or %g\n", fpixel,spixel,stol,test)
            print(f"at %g %g %g\n", pixel_pos[1],pixel_pos[2],pixel_pos[3])
            print(f"hkl= %f %f %f  hkl0= %d %d %d\n", h,k,l,h0,k0,l0)
            print(f" F_cell=%g  F_latt=%g   I = %g\n", F_cell,F_latt,I)
            print(f"I/steps %15.10g\n", I/steps)
            print(f"cap frac   %f\n", capture_fraction)
            print(f"polar   %15.10g\n", polar)
            print(f"omega   %15.10g\n", omega_pixel)
            print(f"pixel   %15.10g\n", raw_pixels[spixel,fpixel])
            print(f"real-space cell vectors (Angstrom):\n")
            print(f"     %-10s  %-10s  %-10s\n","a","b","c")
            print(f"X: %11.8f %11.8f %11.8f\n",a[1]*1e10,b[1]*1e10,c[1]*1e10)
            print(f"Y: %11.8f %11.8f %11.8f\n",a[2]*1e10,b[2]*1e10,c[2]*1e10)
            print(f"Z: %11.8f %11.8f %11.8f\n",a[3]*1e10,b[3]*1e10,c[3]*1e10)
            SCITBX_EXAMINE(fluence)
            SCITBX_EXAMINE(source_I[0])
            SCITBX_EXAMINE(spot_scale)
            SCITBX_EXAMINE(Na)
            SCITBX_EXAMINE(Nb)
            SCITBX_EXAMINE(Nc)
            SCITBX_EXAMINE(airpath)
            SCITBX_EXAMINE(Fclose)
            SCITBX_EXAMINE(Sclose)
            SCITBX_EXAMINE(close_distance)
            SCITBX_EXAMINE(pix0_vector[0])
            SCITBX_EXAMINE(pix0_vector[1])
            SCITBX_EXAMINE(pix0_vector[2])
            SCITBX_EXAMINE(pix0_vector[3])
            SCITBX_EXAMINE(odet_vector[0])
            SCITBX_EXAMINE(odet_vector[1])
            SCITBX_EXAMINE(odet_vector[2])
            SCITBX_EXAMINE(odet_vector[3])
    
    else:
        if(progress_meter and verbose and progress_pixels/100 > 0):
        
            if(progress_pixel % ( progress_pixels/20 ) == 0 or
                ((10*progress_pixel<progress_pixels or
                    10*progress_pixel>9*progress_pixels) and
                (progress_pixel % (progress_pixels/100) == 0))):
            
                print(f"%lu%% done\n",progress_pixel*100/progress_pixels)
            
        
        progress_pixel+=1