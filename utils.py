import numpy as np

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
    