import numpy as np

# Thomson cross section ((e^2)/(4*PI*epsilon0*m*c^2))^2
r_e_sqr = 7.94079248018965e-30

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


def unitize(vector):
    """make provided vector a unit vector"""

    # measure the magnitude
    mag = magnitude(vector)
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

def dot_product(x, y):
    """vector inner product where vector magnitude is 0th element"""
    return x[1]*y[1]+x[2]*y[2]+x[3]*y[3]

def cross_product(x,y):
    z = np.zeros(4)

    """vector cross product where vector magnitude is 0th element"""
    z[1] = x[2]*y[3] - x[3]*y[2]
    z[2] = x[3]*y[1] - x[1]*y[3]
    z[3] = x[1]*y[2] - x[2]*y[1]
    z[0] = 0.0

    return z


def sinc3( x):
    """Fourier transform of a sphere"""
    if(x==0.0): 
        return 1.0
    else:
        return 3.0*(np.sin(x)/x-np.cos(x))/(x*x)


def sincg(x, N):
    """Fourier transform of a grating"""
    if(x==0.0):
        return N
    else:
        return np.sin(x*N)/np.sin(x)

def vector_scale(vector, scale):
    new_vector = np.zeros(4)

    """scale magnitude of provided vector"""
    new_vector[1] = scale*vector[1]
    new_vector[2] = scale*vector[2]
    new_vector[3] = scale*vector[3]

    return magnitude(new_vector)

def magnitude(vector):
    """measure magnitude of provided vector"""
    magn = np.sqrt(vector[1]*vector[1]+vector[2]*vector[2]+vector[3]*vector[3])

    return magn



def rotate_umat(v, umat):
    """
    rotate a vector using a 9-element unitary matrix
    umat has 9 elements
    """

    newv = np.zeros(4)
    
    # for convenience, assign matrix x-y coordinate
    uxx = umat[0];
    uxy = umat[1];
    uxz = umat[2];
    uyx = umat[3];
    uyy = umat[4];
    uyz = umat[5];
    uzx = umat[6];
    uzy = umat[7];
    uzz = umat[8];

    # rotate the vector (x=1,y=2,z=3)
    newv[1] = uxx*v[1] + uxy*v[2] + uxz*v[3]
    newv[2] = uyx*v[1] + uyy*v[2] + uyz*v[3]
    newv[3] = uzx*v[1] + uzy*v[2] + uzz*v[3]

    return newv



def polarization_factor(kahn_factor, incident, diffracted, axis):
    """polarization factor"""
    E_out = np.zeros(4)
    B_out = np.zeros(4)
    psi = 0.0
    
    # unitize the vectors
    _, incident = unitize(incident)
    _, diffracted = unitize(diffracted)
    _, axis = unitize(axis)

    # component of diffracted unit vector along incident beam unit vector
    cos2theta = dot_product(incident,diffracted)
    cos2theta_sqr = cos2theta*cos2theta
    sin2theta_sqr = 1-cos2theta_sqr

    if(kahn_factor != 0.0):
        # tricky bit here is deciding which direciton the E-vector lies in for each source
        # here we assume it is closest to the "axis" defined above

        # cross product to get "vertical" axis that is orthogonal to the cannonical "polarization"
        B_in = cross_product(axis,incident)
        # make it a unit vector
        _, B_in = unitize(B_in)

        # cross product with incident beam to get E-vector direction
        E_in = cross_product(incident,B_in)

        # make it a unit vector
        E_in = unitize(E_in)

        # get components of diffracted ray projected onto the E-B plane
        E_out[0] = dot_product(diffracted,E_in)
        B_out[0] = dot_product(diffracted,B_in)

        # compute the angle of the diffracted ray projected onto the incident E-B plane
        psi = -np.arctan2(B_out[0],E_out[0])
    

    # correction for polarized incident beam
    return 0.5*(1.0 + cos2theta_sqr - kahn_factor*np.cos(2*psi)*sin2theta_sqr)

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

def polint(xa, ya, x):
    x0 = (x-xa[1])*(x-xa[2])*(x-xa[3])*ya[0]/((xa[0]-xa[1])*(xa[0]-xa[2])*(xa[0]-xa[3]))
    x1 = (x-xa[0])*(x-xa[2])*(x-xa[3])*ya[1]/((xa[1]-xa[0])*(xa[1]-xa[2])*(xa[1]-xa[3]))
    x2 = (x-xa[0])*(x-xa[1])*(x-xa[3])*ya[2]/((xa[2]-xa[0])*(xa[2]-xa[1])*(xa[2]-xa[3]))
    x3 = (x-xa[0])*(x-xa[1])*(x-xa[2])*ya[3]/((xa[3]-xa[0])*(xa[3]-xa[1])*(xa[3]-xa[2]))
    y = x0+x1+x2+x3
    return y

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
    

def interpolate_unit_cell():
    """Calculates Fcell"""
    # NOT IMPLEMENTED
    return

    """ 
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
    """
    
