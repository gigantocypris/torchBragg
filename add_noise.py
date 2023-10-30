// function to add different types of noise to the image
void
nanoBragg::add_noise()
{
    encapsulated_twodev image_deviates;
    encapsulated_twodev pixel_deviates;
    int i = 0;
    long cseed;

    double expected_photons,observed_photons,adu;
    /* refer to raw_pixels pixel data */
    floatimage = raw_pixels.begin();

    /* don't bother with this loop if calibration is perfect
     NOTE: applying calibration before Poisson noise simulates loss of photons before the detector
     NOTE: applying calibration after Poisson noise simulates systematics in read-out electronics
     here we do the latter */

    /* re-start the RNG */
    seed = -labs(seed);

    if(verbose) printf("applying calibration at %g%%, flicker noise at %g%%\n",calibration_noise*100.,flicker_noise*100.);
    sum = max_I = 0.0;
    i = sumn = 0;
    for(spixel=0;spixel<spixels;++spixel)
    {
        for(fpixel=0;fpixel<fpixels;++fpixel)
        {

            /* allow for just one part of detector to be rendered */
            if(fpixel < roi_xmin || fpixel > roi_xmax || spixel < roi_ymin || spixel > roi_ymax)
            {
                ++i; continue;
            }
            /* allow for the use of a mask */
            if(maskimage != NULL)
            {
                /* skip any flagged pixels in the mask */
                if(maskimage[i] == 0)
                {
                    ++i; continue;
                }
            }

            /* take input image to be ideal photons/pixel */
            expected_photons = floatimage[i];

            /* negative photons should be taken as invalid? */
            if(expected_photons < 0.0)
            {
                ++i; continue;
            }

            /* simulate 1/f noise in source */
            if(flicker_noise > 0.0){
                expected_photons *= ( 1.0 + flicker_noise * image_deviates.gaussdev( &seed ) );
            }
            /* simulate photon-counting error */
            observed_photons = image_deviates.poidev( expected_photons, &seed );

            /* now we overwrite the flex array, it is now observed, rather than expected photons */
            floatimage[i] = observed_photons;

            /* accumulate number of photons, and keep track of max */
            if(floatimage[i] > max_I) {
                max_I = floatimage[i];
                max_I_x = fpixel;
                max_I_y = spixel;
            }
            sum += observed_photons;
            ++sumn;

            ++i;
        }
    }
    if(verbose) printf("%.0f photons generated on noise image, max= %f at ( %.0f, %.0f )\n",sum,max_I,max_I_x,max_I_y);



    if(calibration_noise > 0.0)
    {
        /* calibration is same from shot to shot, so use well-known seed */
        cseed = -labs(calib_seed);
        sum = max_I = 0.0;
        i = sumn = 0;
        for(spixel=0;spixel<spixels;++spixel)
        {
            for(fpixel=0;fpixel<fpixels;++fpixel)
            {
                /* allow for just one part of detector to be rendered */
                if(fpixel < roi_xmin || fpixel > roi_xmax || spixel < roi_ymin || spixel > roi_ymax)
                {
                    ++i; continue;
                }
                /* allow for the use of a mask */
                if(maskimage != NULL)
                {
                    /* skip any flagged pixels in the mask */
                    if(maskimage[i] == 0)
                    {
                        ++i; continue;
                    }
                }

                /* calibration is same from shot to shot, but varies from pixel to pixel */
                floatimage[i] *= ( 1.0 + calibration_noise * pixel_deviates.gaussdev( &cseed ) );

                /* accumulate number of photons, and keep track of max */
                if(floatimage[i] > max_I) {
                    max_I = floatimage[i];
                    max_I_x = fpixel;
                    max_I_y = spixel;
                }
                sum += floatimage[i];
                ++sumn;

                ++i;
            }
        }
    }
    if(verbose) printf("%.0f photons after calibration error, max= %f at ( %.0f, %.0f )\n",sum,max_I,max_I_x,max_I_y);


    /* now would be a good time to implement PSF?  before we add read-out noise */

    /* now that we have photon count at each point, implement any PSF */
    if(psf_type != UNKNOWN && psf_fwhm > 0.0)
    {
        /* report on sum before the PSF is applied */
        if(verbose) printf("%.0f photons on noise image before PSF\n",sum);
        /* start with a clean slate */
        if(verbose) printf("  applying PSF width = %g um\n",psf_fwhm*1e6);

        apply_psf(psf_type, psf_fwhm/pixel_size, 0);

        /* the flex array is now the blurred version of itself, ready for read-out noise */
    }


    if(verbose) printf("adu = quantum_gain= %g * observed_photons + offset= %g + readout_noise= %g\n",quantum_gain,adc_offset,readout_noise);
    sum = max_I = 0.0;
    i = sumn = 0;
    for(spixel=0;spixel<spixels;++spixel)
    {
        for(fpixel=0;fpixel<fpixels;++fpixel)
        {
            /* allow for just one part of detector to be rendered */
            if(fpixel < roi_xmin || fpixel > roi_xmax || spixel < roi_ymin || spixel > roi_ymax)
            {
                ++i; continue;
            }
            /* allow for the use of a mask */
            if(maskimage != NULL)
            {
                /* skip any flagged pixels in the mask */
                if(maskimage[i] == 0)
                {
                    ++i; continue;
                }
            }

                /* convert photon signal to pixel units */
                adu = floatimage[i]*quantum_gain + adc_offset;

                /* readout noise is in pixel units (adu) */
                if(readout_noise > 0.0){
                    adu += readout_noise * image_deviates.gaussdev( &seed );
            }

            /* once again, overwriting flex array, this time in ADU units */
            floatimage[i] = adu;

            if(adu > max_I) {
                max_I = adu;
                max_I_x = fpixel;
                max_I_y = spixel;
            }
            sum += adu;
            ++sumn;
            ++i;
        }
    }
    if(verbose) printf("%.0f net adu generated on final image, max= %f at ( %.0f, %.0f )\n",sum-adc_offset*sumn,max_I,max_I_x,max_I_y);
}
// end of add_noise()
