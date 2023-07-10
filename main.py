import torch
from simtbx.nanoBragg import nanoBragg
import numpy as np

# make a nanoBragg object

# extract the parameters from the nanoBragg object

# run add_nanoBragg_spots

####

SIM = nanoBragg()


def rotate_axis(v, axis, phi):
    """rotate a point about a unit vector axis"""
    sinphi = np.sin(phi)
    cosphi = np.cos(phi)
    dot = (axis[1]*v[1]+axis[2]*v[2]+axis[3]*v[3])*(1.0-cosphi)
    newv = np.zeros(4)

    newv[1] = axis[1]*dot+v[1]*cosphi+(-axis[3]*v[2]+axis[2]*v[3])*sinphi
    newv[2] = axis[2]*dot+v[2]*cosphi+(+axis[3]*v[1]-axis[1]*v[3])*sinphi
    newv[3] = axis[3]*dot+v[3]*cosphi+(-axis[2]*v[1]+axis[1]*v[2])*sinphi

    return newv


def unitize(vector, new_unit_vector):
    """make provided vector a unit vector"""

    # measure the magnitude
    mag = np.abs(vector)
    new_unit_vector = np.zeros(4)
    if(mag != 0.0):
        # normalize it
        new_unit_vector[1]=vector[1]/mag
        new_unit_vector[2]=vector[2]/mag
        new_unit_vector[3]=vector[3]/mag

    else:
        # can't normalize, report zero vector
        new_unit_vector[0] = 0.0;
        new_unit_vector[1] = 0.0;
        new_unit_vector[2] = 0.0;
        new_unit_vector[3] = 0.0;

    return mag, new_unit_vector


def add_nanoBragg_spot(spixels, 
                       fpixels,
                       phisteps,
                       mosaic_domains,
                       oversample,
                       pixel_size,
                       verbose=True):
    max_I = 0.0
    i = 0
    raw_pixels = torch.zeros((spixels,fpixels))

    # make sure we are normalizing with the right number of sub-steps
    steps = phisteps*mosaic_domains*oversample*oversample
    subpixel_size = pixel_size/oversample

    sum = sumsqr = 0.0
    i = sumn = 0
    progress_pixel = 0
    omega_sum = 0.0

    for spixel in range(spixels):
        loop_over_spixels(spixel,
                          fpixels,
                          raw_pixels,
                          )
    
    if(verbose):
        print("done with pixel loop\n")
        print("solid angle subtended by detector = %g steradian ( %g%% sphere)\n",omega_sum/steps,100*omega_sum/steps/4/np.pi)
        print("max_I= %g sum= %g avg= %g\n",max_I,sum,sum/sumn)


def loop_over_spixels(spixel,
                      fpixels,
                      raw_pixels,
                      ,
                      ):
    for fpixel in range(fpixels):
        loop_over_fpixels(fpixel)


def loop_over_fpixels(oversample,
                      maskimage):

    # allow for just one part of detector to be rendered
    if(fpixel < roi_xmin or fpixel > roi_xmax or spixel < roi_ymin or spixel > roi_ymax):
        # out-of-bounds, move on to next pixel
        i += 1
    # allow for the use of a mask
    elif(maskimage != NULL):
        # skip any flagged pixels in the mask
        if(maskimage[i] == 0):
            i+=1
    else: # render pixel
        # reset photon count for this pixel
        I = 0

        # loop over sub-pixels
        for subS in range(oversample):
            loop_over_ssubpixels(subS)
        # end of sub-pixel x loop

        raw_pixels[spixel,fpixel] += r_e_sqr*fluence*spot_scale*polar*I/steps
        
        # floatimage[i] += r_e_sqr*fluence*spot_scale*polar*I/steps;

        if(floatimage[i] > max_I) {
            max_I = floatimage[i];
            max_I_x = Fdet;
            max_I_y = Sdet;
        }
        sum += floatimage[i];
        sumsqr += floatimage[i]*floatimage[i];
        ++sumn;

        if( printout )
        {
            if((fpixel==printout_fpixel && spixel==printout_spixel) || printout_fpixel < 0)
            {
                twotheta = atan2(sqrt(pixel_pos[2]*pixel_pos[2]+pixel_pos[3]*pixel_pos[3]),pixel_pos[1]);
                test = sin(twotheta/2.0)/(lambda0*1e10);
                printf("%4d %4d : stol = %g or %g\n", fpixel,spixel,stol,test);
                printf("at %g %g %g\n", pixel_pos[1],pixel_pos[2],pixel_pos[3]);
                printf("hkl= %f %f %f  hkl0= %d %d %d\n", h,k,l,h0,k0,l0);
                printf(" F_cell=%g  F_latt=%g   I = %g\n", F_cell,F_latt,I);
                printf("I/steps %15.10g\n", I/steps);
                printf("cap frac   %f\n", capture_fraction);
                printf("polar   %15.10g\n", polar);
                printf("omega   %15.10g\n", omega_pixel);
                printf("pixel   %15.10g\n", floatimage[i]);
                printf("real-space cell vectors (Angstrom):\n");
                printf("     %-10s  %-10s  %-10s\n","a","b","c");
                printf("X: %11.8f %11.8f %11.8f\n",a[1]*1e10,b[1]*1e10,c[1]*1e10);
                printf("Y: %11.8f %11.8f %11.8f\n",a[2]*1e10,b[2]*1e10,c[2]*1e10);
                printf("Z: %11.8f %11.8f %11.8f\n",a[3]*1e10,b[3]*1e10,c[3]*1e10);
                SCITBX_EXAMINE(fluence);
                SCITBX_EXAMINE(source_I[0]);
                SCITBX_EXAMINE(spot_scale);
                SCITBX_EXAMINE(Na);
                SCITBX_EXAMINE(Nb);
                SCITBX_EXAMINE(Nc);
                SCITBX_EXAMINE(airpath);
                SCITBX_EXAMINE(Fclose);
                SCITBX_EXAMINE(Sclose);
                SCITBX_EXAMINE(close_distance);
                SCITBX_EXAMINE(pix0_vector[0]);
                SCITBX_EXAMINE(pix0_vector[1]);
                SCITBX_EXAMINE(pix0_vector[2]);
                SCITBX_EXAMINE(pix0_vector[3]);
                SCITBX_EXAMINE(odet_vector[0]);
                SCITBX_EXAMINE(odet_vector[1]);
                SCITBX_EXAMINE(odet_vector[2]);
                SCITBX_EXAMINE(odet_vector[3]);
            }
        }
        else
        {
            if(progress_meter && verbose && progress_pixels/100 > 0)
            {
                if(progress_pixel % ( progress_pixels/20 ) == 0 ||
                    ((10*progress_pixel<progress_pixels ||
                        10*progress_pixel>9*progress_pixels) &&
                    (progress_pixel % (progress_pixels/100) == 0)))
                {
                    printf("%lu%% done\n",progress_pixel*100/progress_pixels);
                }
            }
            ++progress_pixel;
        }
        ++i;


def loop_over_ssubpixels(oversample):
    for subF in range(oversample):
        loop_over_fsubpixels(subF)


def loop_over_fsubpixels(subpixel_size,
                         fpixel,
                         spixel,
                         oversample,
                         subF,
                         subS,
                         detector_thicksteps,
                         ):

    # absolute mm position on detector (relative to its origin)
    Fdet = subpixel_size*(fpixel*oversample + subF ) + subpixel_size/2.0;
    Sdet = subpixel_size*(spixel*oversample + subS ) + subpixel_size/2.0;


    for thick_tic in range(detector_thicksteps):
        loop_over_detector_thicksteps(thick_tic)



def loop_over_detector_thicksteps():
    # assume "distance" is to the front of the detector sensor layer
    Odet = thick_tic*detector_thickstep

    # construct detector subpixel position in 3D space
    pixel_pos = np.zeros(4)
    pixel_pos[1] = Fdet*fdet_vector[1]+Sdet*sdet_vector[1]+Odet*odet_vector[1]+pix0_vector[1]
    pixel_pos[2] = Fdet*fdet_vector[2]+Sdet*sdet_vector[2]+Odet*odet_vector[2]+pix0_vector[2]
    pixel_pos[3] = Fdet*fdet_vector[3]+Sdet*sdet_vector[3]+Odet*odet_vector[3]+pix0_vector[3]
    pixel_pos[0] = 0.0
    if curved_detector:
        # construct detector pixel that is always "distance" from the sample
        vector[1] = distance*beam_vector[1]
        vector[2] = distance*beam_vector[2]
        vector[3] = distance*beam_vector[3]
        
        # treat detector pixel coordinates as radians
        newvector = rotate_axis(vector,sdet_vector,pixel_pos[2]/distance)
        pixel_pos = rotate_axis(newvector,fdet_vector,pixel_pos[3]/distance)


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
        loop_over_sources(source)


def loop_over_sources():

    incident = np.zeros(4)
    # retrieve stuff from cache
    incident[1] = -source_X[source]
    incident[2] = -source_Y[source]
    incident[3] = -source_Z[source]
    lambda_0 = source_lambda[source]

    # construct the incident beam unit vector while recovering source distance
    source_path, incident = unitize(incident)

    # construct the scattering vector for this pixel
    scattering[1] = (diffracted[1]-incident[1])/lambda_0
    scattering[2] = (diffracted[2]-incident[2])/lambda_0
    scattering[3] = (diffracted[3]-incident[3])/lambda_0

    # sin(theta)/lambda_0 is half the scattering vector length
    stol = 0.5*np.abs(scattering)

    # rough cut to speed things up when we aren't using whole detector
    if(dmin > 0.0 and stol > 0.0 and dmin > 0.5/stol):
        pass
    else: # sweep over phi angles
        for phi_tic in range(phisteps):
            loop_over_phi()


def loop_over_phi(): # only 1 angle to loop over in XFEL
    phi = phi0 + phistep*phi_tic

    if( phi != 0.0 ):
        # rotate about spindle if neccesary
        rotate_axis(a0,ap,spindle_vector,phi)
        rotate_axis(b0,bp,spindle_vector,phi)
        rotate_axis(c0,cp,spindle_vector,phi)


    # enumerate mosaic domains
    for mos_tic in range(mosaic_domains):
        loop_over_mosaic_domains()



def loop_over_mosaic_domains():
    # apply mosaic rotation after phi rotation
    if(mosaic_spread > 0.0):
        rotate_umat(ap,a,mosaic_umats[mos_tic*9])
        rotate_umat(bp,b,mosaic_umats[mos_tic*9])
        rotate_umat(cp,c,mosaic_umats[mos_tic*9])
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
    h0 = static_cast<int>(ceil(h-0.5))
    k0 = static_cast<int>(ceil(k-0.5))
    l0 = static_cast<int>(ceil(l-0.5))

    """
    # structure factor of the lattice (paralelpiped crystal)
    F_latt = sin(M_PI*Na*h)*sin(M_PI*Nb*k)*sin(M_PI*Nc*l)/sin(M_PI*h)/sin(M_PI*k)/sin(M_PI*l)
    """

    # structure factor of the lattice
    F_latt = 1.0
        if(xtal_shape == SQUARE)
    {
        /* xtal is a paralelpiped */
        if(Na>1){
            F_latt *= sincg(M_PI*h,Na);
        }
        if(Nb>1){
            F_latt *= sincg(M_PI*k,Nb);
        }
        if(Nc>1){
            F_latt *= sincg(M_PI*l,Nc);
        }
    }
        else
        {
            /* handy radius in reciprocal space, squared */
            hrad_sqr = (h-h0)*(h-h0)*Na*Na + (k-k0)*(k-k0)*Nb*Nb + (l-l0)*(l-l0)*Nc*Nc ;
        }
        if(xtal_shape == ROUND)
        {
            /* use sinc3 for elliptical xtal shape,
                correcting for sqrt of volume ratio between cube and sphere */
            F_latt = Na*Nb*Nc*0.723601254558268*sinc3(M_PI*sqrt( hrad_sqr * fudge ) );
        }
        if(xtal_shape == GAUSS)
        {
            /* fudge the radius so that volume and FWHM are similar to square_xtal spots */
            F_latt = Na*Nb*Nc*exp(-( hrad_sqr / 0.63 * fudge ));
        }
        if (xtal_shape == GAUSS_ARGCHK)
        {
            /* fudge the radius so that volume and FWHM are similar to square_xtal spots */
            double my_arg = hrad_sqr / 0.63 * fudge; // pre-calculate to check for no Bragg signal
            if (my_arg<35.){ F_latt = Na * Nb * Nc * exp(-(my_arg));}
            else { F_latt = 0.; } // not expected to give performance gain on optimized C++, only on GPU
        }

        if(xtal_shape == TOPHAT)
        {
            /* make a flat-top spot of same height and volume as square_xtal spots */
            F_latt = Na*Nb*Nc*(hrad_sqr*fudge < 0.3969 );
    }
    /* no need to go further if result will be zero */
    if(F_latt == 0.0) continue;


    /* find nearest point on Ewald sphere surface? */
    if( integral_form )
    {

        if( phi != 0.0 || mos_tic > 0 )
        {
            /* need to re-calculate reciprocal matrix */

            /* various cross products */
            cross_product(a,b,a_cross_b);
            cross_product(b,c,b_cross_c);
            cross_product(c,a,c_cross_a);

            /* new reciprocal-space cell vectors */
            vector_scale(b_cross_c,a_star,1e20/V_cell);
            vector_scale(c_cross_a,b_star,1e20/V_cell);
            vector_scale(a_cross_b,c_star,1e20/V_cell);
        }

        /* reciprocal-space coordinates of nearest relp */
        relp[1] = h0*a_star[1] + k0*b_star[1] + l0*c_star[1];
        relp[2] = h0*a_star[2] + k0*b_star[2] + l0*c_star[2];
        relp[3] = h0*a_star[3] + k0*b_star[3] + l0*c_star[3];
//                                  d_star = magnitude(relp)

        /* reciprocal-space coordinates of center of Ewald sphere */
        Ewald0[1] = -incident[1]/lambda/1e10;
        Ewald0[2] = -incident[2]/lambda/1e10;
        Ewald0[3] = -incident[3]/lambda/1e10;
//                                  1/lambda = magnitude(Ewald0)

            /* distance from Ewald sphere in lambda=1 units */
        vector[1] = relp[1]-Ewald0[1];
        vector[2] = relp[2]-Ewald0[2];
        vector[3] = relp[3]-Ewald0[3];
        d_r = magnitude(vector)-1.0;

        /* unit vector of diffracted ray through relp */
        unitize(vector,diffracted0);

        /* intersection with detector plane */
        xd = dot_product(fdet_vector,diffracted0);
        yd = dot_product(sdet_vector,diffracted0);
        zd = dot_product(odet_vector,diffracted0);

        /* where does the central direct-beam hit */
        xd0 = dot_product(fdet_vector,incident);
        yd0 = dot_product(sdet_vector,incident);
        zd0 = dot_product(odet_vector,incident);

        /* convert to mm coordinates */
        Fdet0 = distance*(xd/zd) + Xbeam;
        Sdet0 = distance*(yd/zd) + Ybeam;

        if(verbose>8) printf("integral_form: %g %g   %g %g\n",Fdet,Sdet,Fdet0,Sdet0);
        test = exp(-( (Fdet-Fdet0)*(Fdet-Fdet0)+(Sdet-Sdet0)*(Sdet-Sdet0) + d_r*d_r )/1e-8);
    } // end of integral form


    /* structure factor of the unit cell */
    if(interpolate){
        h0_flr = static_cast<int>(floor(h));
        k0_flr = static_cast<int>(floor(k));
        l0_flr = static_cast<int>(floor(l));


        if ( ((h-h_min+3)>h_range) ||
                (h-2<h_min)           ||
                ((k-k_min+3)>k_range) ||
                (k-2<k_min)           ||
                ((l-l_min+3)>l_range) ||
                (l-2<l_min)  ) {
            if(babble){
                babble=0;
                if(verbose) printf ("WARNING: out of range for three point interpolation: h,k,l,h0,k0,l0: %g,%g,%g,%d,%d,%d \n", h,k,l,h0,k0,l0);
                if(verbose) printf("WARNING: further warnings will not be printed! ");
            }
            F_cell = default_F;
            interpolate=0;
            continue;
        }

        /* integer versions of nearest HKL indicies */
        h_interp[0]=h0_flr-1;
        h_interp[1]=h0_flr;
        h_interp[2]=h0_flr+1;
        h_interp[3]=h0_flr+2;
        k_interp[0]=k0_flr-1;
        k_interp[1]=k0_flr;
        k_interp[2]=k0_flr+1;
        k_interp[3]=k0_flr+2;
        l_interp[0]=l0_flr-1;
        l_interp[1]=l0_flr;
        l_interp[2]=l0_flr+1;
        l_interp[3]=l0_flr+2;

        /* polin function needs doubles */
        h_interp_d[0] = (double) h_interp[0];
        h_interp_d[1] = (double) h_interp[1];
        h_interp_d[2] = (double) h_interp[2];
        h_interp_d[3] = (double) h_interp[3];
        k_interp_d[0] = (double) k_interp[0];
        k_interp_d[1] = (double) k_interp[1];
        k_interp_d[2] = (double) k_interp[2];
        k_interp_d[3] = (double) k_interp[3];
        l_interp_d[0] = (double) l_interp[0];
        l_interp_d[1] = (double) l_interp[1];
        l_interp_d[2] = (double) l_interp[2];
        l_interp_d[3] = (double) l_interp[3];

        /* now populate the "y" values (nearest four structure factors in each direction) */
        for (i1=0;i1<4;i1++) {
            for (i2=0;i2<4;i2++) {
                for (i3=0;i3<4;i3++) {
                        sub_Fhkl[i1][i2][i3]= Fhkl[h_interp[i1]-h_min][k_interp[i2]-k_min][l_interp[i3]-l_min];
                }
            }
            }


        /* run the tricubic polynomial interpolation */
        polin3(h_interp_d,k_interp_d,l_interp_d,sub_Fhkl,h,k,l,&F_cell);
    }

    if(! interpolate)
    {
        if ( (h0<=h_max) && (h0>=h_min) && (k0<=k_max) && (k0>=k_min) && (l0<=l_max) && (l0>=l_min)  ) {
            /* just take nearest-neighbor */
            F_cell = Fhkl[h0-h_min][k0-k_min][l0-l_min];
        }
        else
        {
            F_cell = default_F; // usually zero
        }
    }

    /* now we have the structure factor for this pixel */

    /* polarization factor */
    if(! nopolar){
        /* need to compute polarization factor */
        polar = polarization_factor(polarization,incident,diffracted,polar_vector);
    }
    else
    {
        polar = 1.0;
    }

    /* convert amplitudes into intensity (photons per steradian) */
        I += F_cell*F_cell*F_latt*F_latt*source_I[source]*capture_fraction*omega_pixel;


