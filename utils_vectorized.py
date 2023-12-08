from utils import which_package
import numpy as np
import torch

def Fhkl_remove(Fhkl, h_max, h_min, k_max, k_min, l_max, l_min):
    for key in Fhkl.keys():
        h0 = key[0]
        k0 = key[1]
        l0 = key[2]

        if not((h0<=h_max) and (h0>=h_min) and (k0<=k_max) and (k0>=k_min) and (l0<=l_max) and (l0>=l_min)):
            del Fhkl[key]
    return(Fhkl)


def Fhkl_dict_to_mat(Fhkl, h_max, h_min, k_max, k_min, l_max, l_min, default_F, prefix):
    Fhkl_mat = default_F*prefix.ones([h_max-h_min+1, k_max-k_min+1, l_max-l_min+1])
    for key in Fhkl.keys():
        if int(key[0]) != key[0] or int(key[1]) != key[1] or int(key[2]) != key[2]:
            raise ValueError("hkl indices must be integers")
        h0 = int(key[0])
        k0 = int(key[1])
        l0 = int(key[2])
        Fhkl_mat[h0-h_min, k0-k_min, l0-l_min] = Fhkl[key]
    return(Fhkl_mat)

def sincg_vectorized(x, N, prefix):
    sincg_i = prefix.sin(x*N)/prefix.sin(x)
    sincg_i[x==0] = N
    return sincg_i

def sinc3_vectorized(x, prefix):
    """Fourier transform of a sphere"""
    sinc3_i = 3.0*(prefix.sin(x)/x-prefix.cos(x))/(x*x)
    sinc3_i[x==0] = 1.0
    return sinc3_i

def unitize_vectorized(x, prefix):
    """unitize a vector"""
    x_mag = prefix.sqrt(prefix.sum(x*x, axis=-1))
    x_mag[x_mag==0] = -1

    x_unit = x/x_mag[...,None]
    x_unit[x_mag==-1] = 0
    x_mag[x_mag==-1] = 0
    return x_mag, x_unit

def polarization_factor_vectorized(kahn_factor, incident_mat, diffracted_mat, axis, use_numpy):
    """polarization factor"""
    
    """
    incident_mat is num_sources x 3
    diffracted_mat is x_subpixels x y_subpixels x num_thicknesses x 3

    To add sources dimension to diffracted_mat --> diffracted_mat[:,:,:,None,:]
    """

    prefix, new_array = which_package(use_numpy)
    
    # unitize the vectors
    _, incident_mat = unitize_vectorized(incident_mat, prefix)
    _, diffracted_mat = unitize_vectorized(diffracted_mat, prefix)
    _, axis = unitize_vectorized(axis[None,:], prefix)

    # component of diffracted unit vector along incident beam unit vector
    cos2theta = prefix.sum(incident_mat[None,None,None,:,:]*diffracted_mat[...,None,:], axis=-1)
    cos2theta_sqr = cos2theta*cos2theta
    sin2theta_sqr = 1-cos2theta_sqr

    if(kahn_factor != 0.0):
        # tricky bit here is deciding which direciton the E-vector lies in for each source
        # here we assume it is closest to the "axis" defined above

        # cross product to get "vertical" axis that is orthogonal to the cannonical "polarization"
        B_in = prefix.cross(axis,incident_mat)
        # make it a unit vector
        _, B_in = unitize_vectorized(B_in, prefix)

        # cross product with incident beam to get E-vector direction
        E_in = prefix.cross(incident_mat,B_in)

        # make it a unit vector
        E_in_mag, E_in = unitize_vectorized(E_in, prefix)

        # get components of diffracted ray projected onto the E-B plane
        E_out_mag = prefix.sum(diffracted_mat[...,None,:]*E_in[None,None,None,:,:], axis=-1)

        B_out_mag = prefix.sum(diffracted_mat[...,None,:]*B_in[None,None,None,:,:], axis=-1)

        # compute the angle of the diffracted ray projected onto the incident E-B plane
        if use_numpy:
            atan2 = np.arctan2
        else:
            atan2 = torch.atan2
        psi = -atan2(B_out_mag,E_out_mag)
    else:
        psi = prefix.zeros_like(cos2theta_sqr)

    # correction for polarized incident beam
    return 0.5*(1.0 + cos2theta_sqr - kahn_factor*prefix.cos(2*psi)*sin2theta_sqr)