def add_background(oversample, override_source):
{
    int i;
    int source_start = 0;
    int orig_sources = this->sources;
    int end_sources = this->sources;
    max_I = 0.0;
    floatimage = raw_pixels.begin();
//    double* floatimage(raw_pixels.begin());
//    floatimage = (double *) calloc(spixels*fpixels+10,sizeof(double));

    /* allow user to override automated oversampling decision at call time with arguments */
    if(oversample<=0) oversample = this->oversample;
    if(oversample<=0) oversample = 1;
    bool have_single_source = false;
    if(override_source>=0) {
        /* user-specified idx_single_source in the argument */
        source_start = override_source;
        end_sources = source_start +1;
        have_single_source = true;
    }

    /* make sure we are normalizing with the right number of sub-steps */
    steps = oversample*oversample;
    subpixel_size = pixel_size/oversample;

    /* sweep over detector */
    sum = sumsqr = 0.0;
    sumn = 0;
    progress_pixel = 0;
    omega_sum = 0.0;
    nearest = 0;
    i = 0;
    for(spixel=0;spixel<spixels;++spixel)
    {
        for(fpixel=0;fpixel<fpixels;++fpixel)
        {
            /* allow for just one part of detector to be rendered */
            if(fpixel < roi_xmin || fpixel > roi_xmax || spixel < roi_ymin || spixel > roi_ymax) {
                invalid_pixel[i] = true;
                ++i; continue;
            }

            /* reset background photon count for this pixel */
            Ibg = 0;

            /* loop over sub-pixels */
            for(subS=0;subS<oversample;++subS){
                for(subF=0;subF<oversample;++subF){

                    /* absolute mm position on detector (relative to its origin) */
                    Fdet = subpixel_size*(fpixel*oversample + subF ) + subpixel_size/2.0;
                    Sdet = subpixel_size*(spixel*oversample + subS ) + subpixel_size/2.0;
//                    Fdet = pixel_size*fpixel;
//                    Sdet = pixel_size*spixel;

                    for(thick_tic=0;thick_tic<detector_thicksteps;++thick_tic)
                    {
                        /* assume "distance" is to the front of the detector sensor layer */
                        Odet = thick_tic*detector_thickstep;

                        /* construct detector pixel position in 3D space */
    //                    pixel_X = distance;
    //                    pixel_Y = Sdet-Ybeam;
    //                    pixel_Z = Fdet-Xbeam;
                        pixel_pos[1] = Fdet*fdet_vector[1]+Sdet*sdet_vector[1]+Odet*odet_vector[1]+pix0_vector[1];
                        pixel_pos[2] = Fdet*fdet_vector[2]+Sdet*sdet_vector[2]+Odet*odet_vector[2]+pix0_vector[2];
                        pixel_pos[3] = Fdet*fdet_vector[3]+Sdet*sdet_vector[3]+Odet*odet_vector[3]+pix0_vector[3];
                        pixel_pos[0] = 0.0;
                        if(curved_detector) {
                            /* construct detector pixel that is always "distance" from the sample */
                            vector[1] = distance*beam_vector[1]; vector[2]=distance*beam_vector[2] ; vector[3]=distance*beam_vector[3];
                            /* treat detector pixel coordinates as radians */
                            rotate_axis(vector,newvector,sdet_vector,pixel_pos[2]/distance);
                            rotate_axis(newvector,pixel_pos,fdet_vector,pixel_pos[3]/distance);
    //                             rotate(vector,pixel_pos,0,pixel_pos[3]/distance,pixel_pos[2]/distance);
                        }
                        /* construct the diffracted-beam unit vector to this pixel */
                        airpath = unitize(pixel_pos,diffracted);

                        /* solid angle subtended by a pixel: (pix/airpath)^2*cos(2theta) */
                        omega_pixel = pixel_size*pixel_size/airpath/airpath*close_distance/airpath;
                        /* option to turn off obliquity effect, inverse-square-law only */
                        if(point_pixel) omega_pixel = 1.0/airpath/airpath;
                        omega_sum += omega_pixel;

                        /* now calculate detector thickness effects */
                        if(detector_thick > 0.0)
                        {
                            /* inverse of effective thickness increase */
                            parallax = dot_product(diffracted,odet_vector);
                            capture_fraction = exp(-thick_tic*detector_thickstep/detector_attnlen/parallax)
                                              -exp(-(thick_tic+1)*detector_thickstep/detector_attnlen/parallax);
                        }
                        else
                        {
                            capture_fraction = 1.0;
                        }

                        /* loop over sources now */
                        for(source=source_start; source < end_sources; ++source){
                            double n_source_scale = (have_single_source) ? orig_sources : source_I[source];

                            /* retrieve stuff from cache */
                            incident[1] = -source_X[source];
                            incident[2] = -source_Y[source];
                            incident[3] = -source_Z[source];
                            lambda = source_lambda[source];

                            /* construct the incident beam unit vector while recovering source distance */
                            source_path = unitize(incident,incident);

                            /* construct the scattering vector for this pixel */
                            scattering[1] = (diffracted[1]-incident[1])/lambda;
                            scattering[2] = (diffracted[2]-incident[2])/lambda;
                            scattering[3] = (diffracted[3]-incident[3])/lambda;

                            /* sin(theta)/lambda is half the scattering vector length */
                            stol = 0.5*magnitude(scattering);

                            /* now we need to find the nearest four "stol file" points */
                            while(stol > stol_of[nearest] && nearest <= stols){++nearest; };
                            while(stol < stol_of[nearest] && nearest >= 2){--nearest; };

                            /* cubic spline interpolation */
                            polint(stol_of+nearest-1, Fbg_of+nearest-1, stol, &Fbg);

                            /* allow negative F values to yield negative intensities */
                            sign=1.0;
                            if(Fbg<0.0) sign=-1.0;

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

                            /* accumulate unscaled pixel intensity from this */
                            Ibg += sign*Fbg*Fbg*polar*omega_pixel*capture_fraction*n_source_scale;
                            if(verbose>7 && i==1)printf("DEBUG: Fbg= %g polar= %g omega_pixel= %g source[%d]= %g capture_fraction= %g\n",
                                                           Fbg,polar,omega_pixel,source,source_I[source],capture_fraction);
                        }
                        /* end of source loop */
                    }
                    /* end of detector thickness loop */
                }
                /* end of sub-pixel y loop */
            }
            /* end of sub-pixel x loop */


            /* save photons/pixel (if fluence specified), or F^2/omega if no fluence given */
            floatimage[i] += Ibg*r_e_sqr*fluence*amorphous_molecules/steps;

            if(verbose>7 && i==1)printf(
              "DEBUG: Ibg= %g r_e_sqr= %g fluence= %g amorphous_molecules= %g parallax= %g capfrac= %g omega= %g polar= %g steps= %d\n",
                        Ibg,r_e_sqr,fluence,amorphous_molecules,parallax,capture_fraction,omega_pixel,polar,steps);

            /* override: just plot interpolated structure factor at every pixel, useful for making absorption masks */
            if(Fmap_pixel) floatimage[i]= Fbg;

            /* keep track of basic statistics */
            if(floatimage[i] > max_I || i==0) {
                max_I = floatimage[i];
                max_I_x = Fdet;
                max_I_y = Sdet;
            }
            sum += floatimage[i];
            sumsqr += floatimage[i]*floatimage[i];
            ++sumn;

            /* debugging infrastructure */
            if( printout )
            {
                if((fpixel==printout_fpixel && spixel==printout_spixel) || printout_fpixel < 0)
                {
                    twotheta = atan2(sqrt(pixel_pos[2]*pixel_pos[2]+pixel_pos[3]*pixel_pos[3]),pixel_pos[1]);
                    test = sin(twotheta/2.0)/(lambda0*1e10);
                    printf("%4d %4d : stol = %g or %g\n", fpixel,spixel,stol,test);
                    printf(" F=%g    I = %g\n", F,I);
                    printf("I/steps %15.10g\n", I/steps);
                    printf("polar   %15.10g\n", polar);
                    printf("omega   %15.10g\n", omega_pixel);
                    printf("pixel   %15.10g\n", floatimage[i]);
                }
            }
            else
            {
                if(progress_meter && progress_pixels/100 > 0)
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
            /* end progress meter stuff */

            /* never ever forget to increment this */
            ++i;
        }
        /* end fpixel loop */
    }
    /* end spixel loop */

    if(verbose) printf("\nsolid angle subtended by detector = %g steradian ( %g%% sphere)\n",omega_sum/steps,100*omega_sum/steps/4/M_PI);
    if(verbose) printf("max_I= %g @ ( %g, %g) sum= %g avg= %g\n",max_I,max_I_x,max_I_y,sum,sum/sumn);

}
// end of add_background()

