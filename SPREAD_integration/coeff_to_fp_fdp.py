"""
Write PyTorch function that takes in coefficients (as well as input parameters mean_energy, nchannels, channel_width) and outputs fp and fdp

Usage: libtbx.python $MODULES/torchBragg/SPREAD_integration/coeff_to_fp_fdp.py
"""
import torch

"""
params.spectrum.nchannels
params.spectrum.channel_width
params.beam.mean_energy
"""
def create_energy_vec(nchannels=5, mean_energy=6550, channel_width=10):
    centerline = float(nchannels-1)/2.0
    channel_mean_eV = (torch.arange(nchannels) - centerline) * channel_width + mean_energy
    return channel_mean_eV

def func_start_end(x, shift, constant, a, b, c, d, e):
    """
    function describing the fdp curve before and after the bandwidth
    shift is the end/start energy value of the curve and constant is the end/start corresponding fdp value
    shift and constant are determined by the bandwidth curve and are not directly optimizable
    a, b, c, d, e are the optimizable parameters
    """
    return a*(x-shift)**3 + b*(x-shift)**2 + c*(x-shift) + d*(x)**(-1) - d*(shift)**(-1) + e*(x)**(-2) - e*(shift)**(-2) + constant

def convert_coeff(shift, constant, a, b, c, d, e):
    # convert to the form of f*x**(-2) + g*x**(-1) + h*x**0 + i*x**1 + j*x**2 + k*x**3
    f = e
    g = d
    h = -c*shift + b*shift**2 - a*shift**3 - d*(shift)**(-1) - e*(shift)**(-2) + constant
    i = c - b*2*shift + a*3*shift**2
    j = b - a*3*shift
    k = a
    return np.array([f,g,h,i,j,k])
    
def func_bandwidth(x, sampled_pts):
    """
    Cubic splines with natural boundary conditions (i.e. second derivatives at the very beginning and very end are 0)
    """

def evaluate_fdp(energy_vec, coeff):
    pass

def calculate_fp(energy_vec, coeff):
    pass

# unit tests

if __name__ == '__main__':
    print(create_energy_vec())