import torch

# add spots from nanocrystal simulation
def add_nanoBragg_spots(verbose = 0):
    max_I = 0.0
    i = 0
    floatimage = torch.zeros(raw_pixels.size) # what is raw_pixels.begin()?

    if verbose:
        print(f"TESTING sincg(1,1) = {sincg(1,1)}\n") # sincg is another function in nanoBragg.cpp
    
    # make sure we are normalizing with the right amount of sub-steps
    steps = phisteps * mosaic_domains * oversample * oversample # these are constants
    subpixel_size = pixel_size / oversample

    sum = sumsqr = 0.0
    i = sumn = 0
    progress_pixel = 0
    omega_sum = 0.0
    for spixel in range(spixels): # need to define spixels and fpixels
        for fpixel in range(fpixels):
            # allow for just one part of detector to be rendered
            if fpixel < roi_xmin or fpixel > roi_xmax or spixel < roi_ymin or spixel > roi_ymax:
                continue
            if maskimage is not None:
                if maskimage[i] == 0:
                    continue 
            
            # reset photon count for this pixel
            I = 0

            # loop over sub-pixels 
            for subS in range(oversample):
                for subF in range(oversample):
                    # absolute mm position on detector (relative to its origin)
                    Fdet = subpixel_size * (fpixel * oversample + subF) + subpixel_size / 2.0
                    Sdet = subpixel_size * (Spixel * oversample + subS) + subpixel_size / 2.0

                    for thick_tic in range(detector_thicksteps):
                        # assume "distance" is to the front of the detector sensor layer 
                        Odet = thick_tic * detector_thickstep

                        # construct detector subpixel position in 3D space */
                        # pixel_X = distance;
                        # pixel_Y = Sdet-Ybeam;
                        # pixel_Z = Fdet-Xbeam;
                        pixel_pos[1] = Fdet * fdet_vector[1] + Sdet * sdet_vector[1] + Odet * odet_vector[1] + pix0_vector[1]
                        pixel_pos[2] = Fdet * fdet_vector[2] + Sdet * sdet_vector[2] + Odet * odet_vector[2] + pix0_vector[2]
                        pixel_pos[3] = Fdet * fdet_vector[3] + Sdet * sdet_vector[3] + Odet * odet_vector[3] + pix0_vector[3]
                        pixel_pos[0] = 0.0
                        if curved_detector:
                            # construct detector pixel that is always "distance" from the sample
                            vector[1] = distance * beam_vector[1]
                            vector[2] = distance * beam_vector[2]
                            vector[3] = distance * beam_vector[3]
                            # treat detector pixel coordinates as radians
                            rotate_axis(vector, newvector, sdet_vector, pixel_pos[2] / distance)
                            rotate_axis(newvector, pixel_pos, fdet_vector, pixel_pos[3] / distance)
                        
                        if point_pixel:
                            omega_pixel = 1.0 / airpath / airpath
                        
                        # now calculate detector thickness effects
                        if detector_thick > 0.0 and detector_attnlen > 0.0:
                            # inverse of effective thickness increase
                            parallax = torch.dot(diffracted, odet_vector) # assuming these are tensors
                            capture_fraction = torch.exp(-thick_tic * detector_thickstep / detector_attnlen / parallax)
                            capture_fraction -= torch.exp(-(thick_tic + 1) * detector_thickstep / detector_attnlen / parallax)
                        else:
                            capture_fraction = 1.0
                    
                        # Loop over sources
                        for source in range(sources):
                            # Retrieve stuff from cache
                            incident[1] = -source_X[source]
                            incident[2] = -source_Y[source]
                            incident[3] = -source_Z[source]
                            _lambda = source_lambda[source]  # Using an alternative name for lambda

                            # Construct the incident beam unit vector while recovering source distance
                            source_path = unitize(incident, incident)

                            # Construct the scattering vector for this pixel
                            scattering[1] = (diffracted[1] - incident[1]) / _lambda
                            scattering[2] = (diffracted[2] - incident[2]) / _lambda
                            scattering[3] = (diffracted[3] - incident[3]) / _lambda

                            # sin(theta)/lambda is half the scattering vector length
                            stol = 0.5 * torch.linalg.norm(scattering) # magnitude?

                            # Rough cut to speed things up when we aren't using the whole detector
                            if dmin > 0.0 and stol > 0.0:
                                if dmin > 0.5 / stol:
                                    continue
                            
                            # sweep over phi angles 
                            for phi_tic in range(phisteps):
                                phi = phi0 + phistep * phi_tic

                                if phi != 0.0:
                                    # rotate about spindle if necessary
                                    rotate_axis(a0, ap, spindle_vector, phi)
                                    rotate_axis(b0, bp, spindle_vector, phi)
                                    rotate_axis(c0, cp, spindle_vector, phi)

                                # enumerate mosaic domains
                                for mos_tic in range(mosaic_domains):
                                    # apply mosaic rotation after phi rotation
                                    if mosaic_spread > 0.0:
                                        rotate_umat(ap, a, mosaic_umats[mos_tic*9:mos_tic*9+9])
                                        rotate_umat(bp, b, mosaic_umats[mos_tic*9:mos_tic*9+9])
                                        rotate_umat(cp, c, mosaic_umats[mos_tic*9:mos_tic*9+9])
                                    else:
                                        a[1] = ap[1]
                                        a[2] = ap[2]
                                        a[3] = ap[3]
                                        b[1] = bp[1]
                                        b[2] = bp[2]
                                        b[3] = bp[3]
                                        c[1] = cp[1]
                                        c[2] = cp[2]
                                        c[3] = cp[3]

                                    # construct fractional Miller indicies
                                    h = torch.dot(a, scattering)
                                    k = torch.dot(b, scattering)
                                    l = torch.dot(c, scattering)

                                    # round off to nearest whole index
                                    h0 = torch.ceil(h-0.5)
                                    k0 = torch.ceil(k-0.5)
                                    l0 = torch.ceil(l-0.5)

                                    # structure factor of the lattice (paralelpiped crystal)
                                    # F_latt = sin(M_PI*Na*h)*sin(M_PI*Nb*k)*sin(M_PI*Nc*l)/sin(M_PI*h)/sin(M_PI*k)/sin(M_PI*l);
                                    F_latt = 1.0
                                    if xtal_shape == SQUARE:
                                        # xtal is a paralelpiped
                                        if Na > 1:
                                            F_latt *= sincg(M_PI * h, Na)
                                        if Nb> 1:
                                            F_latt *= sincg(M_PI * k, Nb)
                                        if Nc> 1:
                                            F_latt *= sincg(M_PI * l, Nc)
                                    else:
                                        # handy radius in reciprocal space, squared
                                        hrad_sqr = (h - h0) ** 2 * Na ** 2 + (k - k0) ** 2 * Nb ** 2 + (l - l0) ** 2 * Nc ** 2
                                    if xtal_shape == "ROUND":
                                        # use sinc3 for elliptical xtal shape, correcting for sqrt of volume ratio between cube and sphere
                                        F_latt = Na * Nb * Nc * 0.723601254558268 * sinc3(M_PI * torch.sqrt(hrad_sqr * fudge))
                                    if xtal_shape == "GAUSS":
                                        # fudge the radius so that volume and FWHM are similar to square_xtal spots
                                        F_latt = Na * Nb * Nc * torch.exp(-hrad_sqr / 0.63 * fudge)
                                    if xtal_shape == "GAUSS_ARGCHK":
                                        # fudge the radius so that volume and FWHM are similar to square_xtal spots
                                        my_arg = hrad_sqr / 0.63 * fudge # pre-calculate to check for no Bragg signal
                                        if my_arg < 35.:
                                            F_latt = Na * Nb * Nc * torch.exp(-my_arg)
                                        else:
                                            F_latt = torch.tensor(0.0)  # not expected to give performance gain on optimized C++, only on GPU
                                    if xtal_shape == "TOPHAT":
                                        # make a flat-top spot of the same height and volume as square_xtal spots
                                        F_latt = Na * Nb * Nc * (hrad_sqr * fudge < 0.3969)
                                    # no need to go further if result will be zero
                                    if F_latt == 0.0:
                                        continue

                                    # find nearest point on Ewald sphere surface? 
                                    if integral_form:
                                        if phi != 0.0 or mos_tic > 0:
                                            # need to re-calculate reciprocal matrix

                                            # various cross products 
                                            a_cross_b = torch.linalg.cross(a,b)
                                            b_cross_c = torch.linalg.cross(b,c)
                                            c_cross_a = torch.linalg.cross(c,a)

                                            # new reciprocal-space cell vectors
                                            vector_scale(b_cross_c,a_star,1e20/V_cell) # save these variables?
                                            vector_scale(c_cross_a,b_star,1e20/V_cell)
                                            vector_scale(a_cross_b,c_star,1e20/V_cell)
                                        
                                        # reciprocal-space coordinates of nearest relp
                                        relp[1] = h0 * a_star[1] + k0 * b_star[1] + l0 * c_star[1]
                                        relp[2] = h0 * a_star[2] + k0 * b_star[2] + l0 * c_star[2]
                                        relp[3] = h0 * a_star[3] + k0 * b_star[3] + l0 * c_star[3]
                                        # d_star = magnitude(relp)

                                        # reciprocal-space coordinates of center of Ewald sphere
                                        Ewald0[1] = -incident[1] / _lambda / 1e10
                                        Ewald0[1] = -incident[1] / _lambda / 1e10
                                        Ewald0[2] = -incident[2] / _lambda / 1e10
                                        # 1/_lambda = magnitude(Ewald0)

                                        # distance from Ewald sphere in lambda = 1 units
                                        vector[1] = relp[1]-Ewald0[1]
                                        vector[2] = relp[2]-Ewald0[2]
                                        vector[3] = relp[3]-Ewald0[3]
                                        d_r = torch.linalg.norm(vector)-1.0

                                        # unit vector of diffracted ray through relp
                                        vector = unitize(vector, diffracted0) 

                                        # intersection with detector plane
                                        xd = torch.dot(fdet_vector,diffracted0)
                                        yd = torch.dot(sdet_vector,diffracted0)
                                        zd = torch.dot(odet_vector,diffracted0)

                                        # where does the central direct-beam hit 
                                        xd0 = torch.dot(fdet_vector,incident)
                                        yd0 = torch.dot(sdet_vector,incident)
                                        zd0 = torch.dot(odet_vector,incident)

                                        # convert to mm coordinates 
                                        Fdet0 = distance * (xd/zd) + Xbeam
                                        Sdet0 = distance * (yd/zd) + Ybeam

                                        if verbose > 8:
                                            print(f"integral_form: {Fdet} {Sdet}   {Fdet0} {Sdet0}")
                                        test = torch.exp(-((Fdet-Fdet0)*(Fdet-Fdet0)+(Sdet-Sdet0)*(Sdet-Sdet0) + d_r*d_r)/1e-8)

                                    # structure factor of the unit cell
                                    if interpolate:
                                        h0_flr = torch.floor(h).int()
                                        k0_flr = torch.floor(k).int()
                                        l0_flr = torch.floor(l).int()

                                        if ((h - h_min + 3) > h_range) or \
                                        (h - 2 < h_min) or \
                                        ((k - k_min + 3) > k_range) or \
                                        (k - 2 < k_min) or \
                                        ((l - l_min + 3) > l_range) or \
                                        (l - 2 < l_min):
                                            if babble:
                                                babble = 0
                                                if verbose:
                                                    print(f"WARNING: out of range for three point interpolation: h,k,l,h0,k0,l0: {h},{k},{l},{h0},{k0},{l0}")
                                                    print("WARNING: further warnings will not be printed!")
                                            F_cell = default_F
                                            interpolate = 0
                                            continue

                                        # integer versions of nearest HKL indicies
                                        h_interp = torch.tensor([h0_flr - 1, h0_flr, h0_flr + 1, h0_flr + 2])
                                        k_interp = torch.tensor([k0_flr - 1, k0_flr, k0_flr + 1, k0_flr + 2])
                                        l_interp = torch.tensor([l0_flr - 1, l0_flr, l0_flr + 1, l0_flr + 2])

                                        # polin function needs doubles
                                        h_interp_d = h_interp.double()
                                        k_interp_d = k_interp.double()
                                        l_interp_d = l_interp.double()

                                        # now populate the "y" values (nearest four structure factors in each direction)
                                        for i1 in range(4):
                                            for i2 in range(4):
                                                for i3 in range(4):
                                                    sub_Fhkl[i1][i2][i3]= Fhkl[h_interp[i1]-h_min][k_interp[i2]-k_min][l_interp[i3]-l_min]

                                        #run the tricubic polynomial interpolation
                                        polin3(h_interp_d, k_interp_d, l_interp_d, sub_Fhkl, h, k, l, F_cell)

                                    if not interpolate:
                                        if (h0 <= h_max) and (h0 >= h_min) \
                                                and (k0 <= k_max) and (k0 >= k_min) \
                                                and (l0 <= l_max) and (l0 >= l_min):
                                            # just take nearest-neighbor
                                            F_cell = Fhkl[h0 - h_min, k0 - k_min, l0 - l_min]
                                        else:
                                            F_cell = default_F  # usually zero

                                    # now we have the structure factor for this pixel

                                    # polarization factor
                                    if not nopolar:
                                        # need to compute polarization factor
                                        polar = polarization_factor(polarization, incident, diffracted, polar_vector)
                                    else:
                                        polar = 1.0
                                    
                                    # convert amplitudes into intensity (photons per steradian)
                                    I += F_cell * F_cell * F_latt * F_latt * source_I[source] * capture_fraction * omega_pixel
                                # end of mosaic loop
                            # end of phi loop
                        # end of source loop
                    # end of detector thickness loop
                # end of subpixel y loop
            # end of subpixel x loop


            floatimage[i] += r_e_sqr * fluence * spot_scale * polar * I / steps
            if floatimage[i] > max_I:
                max_I = floatimage[i]
                max_I_x = Fdet
                max_I_y = Sdet
            sum += floatimage[i]
            sumsqr += floatimage[i] * floatimage[i]
            sumn += 1

            if printout:
                if (fpixel == printout_fpixel and spixel == printout_spixel) or printout_fpixel < 0:
                    twotheta = atan2(sqrt(pixel_pos[2] * pixel_pos[2] + pixel_pos[3] * pixel_pos[3]), pixel_pos[1])
                    test = sin(twotheta / 2.0) / (lambda0 * 1e10)
                    print(f"{fpixel:4d} {spixel:4d} : stol = {stol:g} or {test:g}")
                    print(f"at {pixel_pos[1]:g} {pixel_pos[2]:g} {pixel_pos[3]:g}")
                    print(f"hkl = {h:f} {k:f} {l:f}  hkl0 = {h0:d} {k0:d} {l0:d}")
                    print(f"F_cell = {F_cell:g}  F_latt = {F_latt:g}   I = {I:g}")
                    print(f"I/steps {I/steps:15.10g}")
                    print(f"cap frac   {capture_fraction:f}")
                    print(f"polar   {polar:15.10g}")
                    print(f"omega   {omega_pixel:15.10g}")
                    print(f"pixel   {floatimage[i]:15.10g}")
                    print("real-space cell vectors (Angstrom):")
                    print("     %-10s  %-10s  %-10s" % ("a", "b", "c"))
                    print(f"X: {a[1]*1e10:11.8f} {b[1]*1e10:11.8f} {c[1]*1e10:11.8f}")
                    print(f"Y: {a[2]*1e10:11.8f} {b[2]*1e10:11.8f} {c[2]*1e10:11.8f}")
                    print(f"Z: {a[3]*1e10:11.8f} {b[3]*1e10:11.8f} {c[3]*1e10:11.8f}")
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
                if progress_meter and verbose and progress_pixels / 100 > 0:
                    if progress_pixel % (progress_pixels / 20) == 0 or (
                        (10 * progress_pixel < progress_pixels or 10 * progress_pixel > 9 * progress_pixels) and
                        (progress_pixel % (progress_pixels / 100) == 0)
                    ):
                        print(f"{progress_pixel*100/progress_pixels:.0f}% done")
                progress_pixel += 1
            i += 1
    
    if verbose:
        print("done with pixel loop")

    if verbose:
        print("solid angle subtended by detector =", omega_sum/steps, "steradian (", 100*omega_sum/steps/4/M_PI, "% sphere)")
        print("max_I =", max_I, "sum =", sum, "avg =", sum/sumn)

# end of add_nanoBragg_spots()