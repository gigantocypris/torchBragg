/* function for applying the PSF, copies over raw pixels with blurred version of itself */
void
nanoBragg::apply_psf(shapetype psf_type, double fwhm_pixels, int user_psf_radius)
{
    max_I=0.0;
    double* inimage(raw_pixels.begin());
    double *outimage=NULL;
    double *kernel;
    int x0,y0,x,y,dx,dy;
    double g,rsq;
    double photon_noise,lost_photons=0.0,total_lost_photons=0.0;
    int maxwidth,kernel_size,psf_radius;
    int i,j,k;
    double photonloss_factor = 10.0; // inverse of maximum tolerable number of lost photons


    /* take the member value for PSF radius if it is set and nothing was in the function call */
    if(user_psf_radius <= 0 && this->psf_radius > 0) user_psf_radius = this->psf_radius;
    /* find a fwhm for the PSF in pixel units */
    if(fwhm_pixels <= 0.0) fwhm_pixels = this->psf_fwhm/this->pixel_size;
    /* update the members, for posterity */
    this->psf_fwhm = fwhm_pixels * this->pixel_size;
    this->psf_radius = user_psf_radius;

    if(verbose>7) printf("apply_psf(): user_psf_radius = %d\n",user_psf_radius);
    if(verbose>7) printf("apply_psf(): updated psf_fwhm = %g  pixel_size= %g\n",psf_fwhm,pixel_size);

    /* convert fwhm to "g" distance : fwhm = sqrt((2**(2./3)-1))/2*g */
    g = fwhm_pixels * 0.652383013252053;

    if(psf_type != GAUSS && psf_type != FIBER)
    {
        if(verbose) printf("ERROR: unknown PSF type\n");
        return;
    }

    pixels = fpixels*spixels;
    if(pixels == 0)
    {
        if(verbose) printf("ERROR: apply_psf image has zero size\n");
        return;
    }

    if(fwhm_pixels <= 0.0)
    {
        if(verbose) printf("WARNING: apply_psf function has zero size\n");
        return;
    }

    /* start with a clean slate */
    if(outimage!=NULL) free(outimage);
    outimage = (double *) calloc(pixels+10,sizeof(double));

    psf_radius = user_psf_radius;
    if(psf_radius <= 0)
    {
        /* auto-select radius */

        /* preliminary stats */
        max_I = 0.0;
        for(i=0;i<pixels;++i)
        {
            /* optionally scale the input file */
            if(max_I < inimage[i]) max_I = inimage[i];
        }
        if(verbose) printf("  maximum input photon/pixel: %g\n",max_I);

        if(max_I<=0.0)
        {
            /* nothing to blur */
            if(verbose) printf("WARNING: no photons, PSF skipped\n");
            return;
        }

        /* at what level will an error in intensity be lost? */
        photon_noise = sqrt(max_I);
        lost_photons = photon_noise/photonloss_factor;
        if(verbose) printf("apply_psf() predicting %g lost photons\n",lost_photons);

        if(psf_type == GAUSS)
        {
            /* calculate the radius beyond which only 0.5 photons will fall */
            psf_radius = 1+ceil( sqrt(-log(lost_photons/max_I)/log(4.0)/2.0)*fwhm_pixels );
            if(verbose) printf("  auto-selected psf_radius = %d x %d pixels for rendering kernel\n",psf_radius,psf_radius);
        }
        if(psf_type == FIBER)
        {
            /* calculate the radius r beyond which only 0.5 photons will fall */
            /* r = sqrt((g*(max_I/0.5))**2-g**2)
                 ~ 2*g*max_I */
            psf_radius = 1+ceil( g*(max_I/lost_photons)  );
            if(verbose) printf("  auto-selected psf_radius = %d x %d pixels for rendering kernel\n",psf_radius,psf_radius);
        }
        if(psf_radius == 0) psf_radius = 1;
    }
    /* limit psf kernel to be no bigger than 4x the input image */
    maxwidth = fpixels;
    if(spixels > maxwidth) maxwidth = spixels;
    if(psf_radius > maxwidth) psf_radius = maxwidth;
    kernel_size = 2*psf_radius+1;
    if(verbose>6) printf("apply_psf() kernel_size= %d\n",kernel_size);

    /* now alocate enough space to store the PSF kernel image */
    kernel = (double *) calloc(kernel_size*kernel_size,sizeof(double));
    if(kernel == NULL)
    {
        perror("apply_psf: could not allocate memory for PSF kernel");
        exit(9);
    }

    /* cache the PSF in an array */
    for(dy=-psf_radius;dy<=psf_radius;++dy)
    {
        for(dx=-psf_radius;dx<=psf_radius;++dx)
        {
            rsq = dx*dx+dy*dy;
            if(rsq > psf_radius*psf_radius) continue;

            /* this could be more efficient */
            k = kernel_size*(kernel_size/2+dy)+kernel_size/2+dx;


            if( psf_type == GAUSS ) {
                kernel[k] = integrate_gauss_over_pixel(dx,dy,fwhm_pixels,1.0);
            }
            if( psf_type == FIBER ) {
                kernel[k] = integrate_fiber_over_pixel(dx,dy,g,1.0);
            }
        }
    }

    /* implement PSF  */
    double sum_in = 0.0, sum_out = 0.0;
    sumn = 0;
    for(i=0;i<pixels;++i)
    {
        x0 = i%fpixels;
        y0 = (i-x0)/fpixels;

        /* skip if there is nothing to add */
        if(inimage[i] <= 0.0) continue;

        sum_in += inimage[i]; ++sumn;

        if(user_psf_radius != 0)
        {
            psf_radius = user_psf_radius;
        }
        else
        {
            /* at what level will an error in intensity be lost? */
            photon_noise = sqrt(inimage[i]);
            lost_photons = photon_noise/photonloss_factor;

            if(psf_type == GAUSS)
            {
                /* calculate the radius beyond which only 0.5 photons will fall
                   r = sqrt(-log(lost_photons/total_photons)/log(4)/2)*fwhm */
                psf_radius = 1+ceil( sqrt(-log(lost_photons/inimage[i])/log(16.0))*fwhm_pixels );
//              printf("  auto-selected psf_radius = %d pixels\n",psf_radius);
            }
            if(psf_type == FIBER)
            {
                /* calculate the radius beyond which only 0.5 photons will fall
                   r = sqrt((g*(total_photons/lost_photons))**2-g**2)
                     ~ g*total_photons/lost_photons */
                psf_radius = 1+ceil( g*(inimage[i]/lost_photons)  );
//              printf("  (%d,%d) auto-selected psf_radius = %d pixels\n",x0,y0,psf_radius);
            }
        }
        if(psf_radius == 0) psf_radius = 1;
        /* limit psf kernel to be no bigger than 4x the input image */
        maxwidth = fpixels;
        if(spixels > maxwidth) maxwidth = spixels;
        if(psf_radius > maxwidth) psf_radius = maxwidth;

        /* given the radius, how many photons will escape? */
        if(psf_type == GAUSS)
        {
            /* r = sqrt(-log(lost_photons/total_photons)/log(16))*fwhm */
            /* lost_photons = total_photons*exp(-log(16)*(r^2/fwhm^2)) */
            rsq = psf_radius;
            rsq = rsq/fwhm_pixels;
            rsq = rsq*rsq;
            lost_photons = inimage[i]*exp(-log(16.0)*rsq);
        }
        if(psf_type == FIBER)
        {
            /* r ~ g*total_photons/lost_photons
               normalized integral from r=inf to "r" :  g/sqrt(g**2+r**2) */
            lost_photons = inimage[i]*g/sqrt(g*g+psf_radius*psf_radius);
        }
        /* accumulate this so we can add it to the whole image */
        total_lost_photons += lost_photons;

        for(dx=-psf_radius;dx<=psf_radius;++dx)
        {
            for(dy=-psf_radius;dy<=psf_radius;++dy)
            {
                /* this could be more efficient */
                k = kernel_size*(kernel_size/2+dy)+kernel_size/2+dx;
                if(kernel[k] == 0.0) continue;

                rsq = dx*dx+dy*dy;
                if(rsq > psf_radius*psf_radius) continue;
                x = x0+dx;
                y = y0+dy;
                if(x<0 || x>fpixels) continue;
                if(y<0 || y>spixels) continue;

                /* index into output array */
                j = y*fpixels+x;
                /* do not wander off the output array */
                if(j<0 || j > pixels) continue;

                outimage[j] += inimage[i]*kernel[k];
            }
        }
    }
    /* now we have some lost photons, add them back "everywhere" */
    lost_photons = total_lost_photons/pixels;
    if(verbose) printf("adding back %g lost photons\n",total_lost_photons);
    for(i=0;i<pixels;++i)
    {
        sum_out += outimage[i];
        outimage[i] += lost_photons;
        if(verbose>7 && i==pixels/2) printf("apply_psf() pixel=%d in= %g out= %g \n",i,inimage[i],outimage[i]);
    }
    if(verbose>7) printf("apply_psf() sum_in=  %g\n",sum_in);
    if(verbose>7) printf("apply_psf() sum_out= %g\n",sum_out);
    if(verbose>7) printf("apply_psf() sum_out= %g (after correction)\n",sum_out+total_lost_photons);

    /* don't need kernel anymore. */
    free(kernel);

    i=pixels/2;
    if(verbose>7) printf("apply_psf() pixel=%d in= %g out= %g \n",i,inimage[i],outimage[i]);

    /* and now.  No idea how to exchange buffers without confusing Python, so lets just copy it back */
    memcpy(inimage,outimage,pixels*sizeof(double));

    free(outimage);
    return;
}
// end of apply_psf()
