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
    fp_full = []
    for energy in energy_vec:
        fp = fdp_fp_integrate(energy, intervals_mat, coeff_mat, powers_mat, relativistic_correction)
        fp_full.append(fp)
    
    return fp_full

def fdp_fp_integrate(energy, intervals_mat, coeff_mat, powers_mat, relativistic_correction):
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
    fp = 0
    for ind, interval in enumerate(intervals_mat):
        energy_start = interval[0]
        energy_end = interval[1]
        coeff = coeff_mat[ind]
        powers = powers_mat[ind]
        fp += 1/(np.pi*energy)*fdp_fp_easy_integral(energy, energy_start, energy_end, coeff, powers)
        fp += -1/(np.pi*energy)*fdp_fp_hard_integral(energy, energy_start, energy_end, coeff, powers)
    return fp + relativistic_correction


def fdp_fp_easy_integral(energy, energy_start, energy_end, coeff, powers):
    """
    Get integral at energy for the term with the x+E denominator
    """

    # Check that powers is sorted in ascending order
    assert all(powers[i] <= powers[i+1] for i in range(len(powers)-1))

    integral = 0
    for ind,n in enumerate(powers):
        coeff_i = coeff[ind]
        if n >= 0:
            for k in range(1,n+2):
                integral += coeff_i*(((-energy)**(n-k+1))/k)*(energy_end**k - energy_start**k)
        integral += coeff_i*((-energy)**(n+1))*torch.log(torch.abs((energy_end + energy)/(energy_start+energy)))
        if n <= -2:
            integral += -coeff_i*((-energy)**(n+1))*torch.log(torch.abs(energy_end/energy_start))
        if n <= -3:
            for k in range(n+2,0):
                integral += coeff_i*(((-energy)**(n-k+1))/(-k))*(energy_end**k - energy_start**k)
    return integral


def fdp_fp_hard_integral(energy, energy_start, energy_end, coeff, powers):
    """
    Get integral at energy for the term with the x-E denominator
    """

    # Check that powers is sorted in ascending order
    assert all(powers[i] <= powers[i+1] for i in range(len(powers)-1))

    integral = 0
    for ind,n in enumerate(powers):
        coeff_i = coeff[ind]
        if n >= 0:
            for k in range(1,n+2):
                integral += coeff_i*((energy**(n-k+1))/k)*(energy_end**k - energy_start**k)
        try:
            integral += coeff_i*(energy**(n+1))*torch.log(torch.abs((energy_end - energy)/(energy_start - energy))) ## Problem term here
        except FloatingPointError as rw:
            print("FloatingPointError:", rw)
            print("Check that energy is not at the endpoints of any interval")
        if n <= -2:
            integral += -coeff_i*(energy**(n+1))*torch.log(torch.abs(energy_end/energy_start))
        if n <= -3:
            for k in range(n+2,0):
                integral += coeff_i*((energy**(n-k+1))/(-k))*(energy_end**k - energy_start**k)
    return integral