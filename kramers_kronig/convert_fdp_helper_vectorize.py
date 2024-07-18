import torch
import numpy as np
from torchBragg.kramers_kronig.convert_fdp_helper import reformat_fdp

def get_physical_params_fp(energy_vec, energy_vec_bandwidth, free_params, coeff_vec_bandwidth, relativistic_correction):
    """
    This function gets the fp physical parameters from free_params

    energy_vec are the energies we want to evaluate at, energy_vec_bandwidth are offset and where the fdp_vec_base values are
    this offset is necessary to evaluate the integral to compute fp_vec at energy_vec energies

    params is in the following form:
    func_start_end       -- func_bandwith  -- func_start_end
    [a0, b0, c0, d0, e0] -- [fdp_vec_base] -- [a1, b1, c1, d1, e1]
    """
    intervals_mat, coeff_mat, powers_mat = reformat_fdp(energy_vec_bandwidth, free_params, coeff_vec_bandwidth)

    # Now convert fdp to fp, account for relativistic correction
    fp_full = fdp_fp_integrate(energy_vec, intervals_mat, coeff_mat, powers_mat, relativistic_correction)
    
    return fp_full

def fdp_fp_integrate(energy_vec, intervals_mat, coeff_mat, powers_mat, relativistic_correction):
    """
    Find fp from fdp using the Kramers-Kronig relation at energy
    energy cannot be at the endpoints of any interval!!
    powers must be sorted in ascending order

    Use the analytical expressions for the integrals from Watts (2014)
    Notation in Watts (2014):
        E is the energy where we want to find fp (energy in this case)
        Degree of polynomial starts at M, M can be negative
        Degree of polynomial ends at N, assume N is positive
        jth interval
    """

    fp = 1/(np.pi*energy_vec)*fdp_fp_easy_integral(energy_vec, intervals_mat, coeff_mat, powers_mat)
    fp += -1/(np.pi*energy_vec)*fdp_fp_hard_integral(energy_vec, intervals_mat, coeff_mat, powers_mat)
    
    return fp + relativistic_correction

def fdp_fp_easy_integral(energy_vec, intervals_mat, coeff_mat, powers_mat):
    """
    Get integrals at energy_vec for the term with the x+E denominator
    variables are converted to the form (energy x interval x power x dummy_k)
    """

    energy_vec = energy_vec[:,None,None,None]
    intervals_start = intervals_mat[:,0][None,:,None,None]
    intervals_end = intervals_mat[:,1][None,:,None,None]
    coeff_mat = coeff_mat[None,:,:,None]
    powers_mat = powers_mat[None,:,:,None]

    integral = coeff_mat*((-energy_vec)**(powers_mat+1))*torch.log(torch.abs((intervals_end + energy_vec)/(intervals_start + energy_vec))) # partial_integral_1

    if torch.max(powers_mat)>=0:
        k = torch.arange(1, torch.max(powers_mat)+2)
        k_mat = k.repeat(intervals_mat.shape[0],powers_mat.shape[2],1)[None]

        partial_integral_0 = coeff_mat*(((-energy_vec)**(powers_mat-k_mat+1))/k_mat)*(intervals_end**k_mat - intervals_start**k_mat)
        partial_integral_0 = partial_integral_0*(powers_mat>=0)*((k_mat - powers_mat)<2)

        integral += torch.sum(partial_integral_0,axis=-1)[:,:,:,None]

    if torch.min(powers_mat)<=-2:
        partial_integral_2 = -coeff_mat*((-energy_vec)**(powers_mat+1))*torch.log(torch.abs(intervals_end/intervals_start))
        partial_integral_2 = partial_integral_2*(powers_mat<=-2)
        integral += partial_integral_2

    if torch.min(powers_mat)<=-3:
        k = torch.arange(torch.min(powers_mat)+2,0)
        k_mat = k.repeat(intervals_mat.shape[0],powers_mat.shape[2],1)[None]
        partial_integral_3 = coeff_mat*(((-energy_vec)**(powers_mat-k_mat+1))/(-k_mat))*(intervals_end**k_mat - intervals_start**k_mat)
        partial_integral_3 = partial_integral_3*(powers_mat<=-3)*((k_mat - powers_mat)>1)
        integral += torch.sum(partial_integral_3,axis=-1)[:,:,:,None]

    return torch.sum(integral,axis=[1,2,3])

def fdp_fp_hard_integral(energy_vec, intervals_mat, coeff_mat, powers_mat):
    """
    Get integrals at energy_vec for the term with the x-E denominator
    variables are converted to the form (energy x interval x power x dummy_k)
    Make sure there are no clashes between energy_vec and intervals_mat
    """

    energy_vec = energy_vec[:,None,None,None]
    intervals_start = intervals_mat[:,0][None,:,None,None]
    intervals_end = intervals_mat[:,1][None,:,None,None]
    coeff_mat = coeff_mat[None,:,:,None]
    powers_mat = powers_mat[None,:,:,None]

    integral = coeff_mat*(energy_vec**(powers_mat+1))*torch.log(torch.abs((intervals_end - energy_vec)/(intervals_start - energy_vec))) # Problem term

    if torch.max(powers_mat)>=0:
        k = torch.arange(1, torch.max(powers_mat)+2)
        k_mat = k.repeat(intervals_mat.shape[0],powers_mat.shape[2],1)[None]

        partial_integral_0 = coeff_mat*((energy_vec**(powers_mat-k_mat+1))/k_mat)*(intervals_end**k_mat - intervals_start**k_mat)
        partial_integral_0 = partial_integral_0*(powers_mat>=0)*((k_mat - powers_mat)<2)

        integral += torch.sum(partial_integral_0,axis=-1)[:,:,:,None]

    if torch.min(powers_mat)<=-2:
        partial_integral_2 = -coeff_mat*(energy_vec**(powers_mat+1))*torch.log(torch.abs(intervals_end/intervals_start))
        partial_integral_2 = partial_integral_2*(powers_mat<=-2)
        integral += partial_integral_2

    if torch.min(powers_mat)<=-3:
        k = torch.arange(torch.min(powers_mat)+2,0)
        k_mat = k.repeat(intervals_mat.shape[0],powers_mat.shape[2],1)[None]
        partial_integral_3 = coeff_mat*((energy_vec**(powers_mat-k_mat+1))/(-k_mat))*(intervals_end**k_mat - intervals_start**k_mat)
        partial_integral_3 = partial_integral_3*(powers_mat<=-3)*((k_mat - powers_mat)>1)
        integral += torch.sum(partial_integral_3,axis=-1)[:,:,:,None]

    return torch.sum(integral,axis=[1,2,3])