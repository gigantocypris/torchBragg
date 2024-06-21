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
    """
    return a*(x-shift)**3 + b*(x-shift)**2 + c*(x-shift) + d*(x)**(-1) - d*(shift)**(-1) + e*(x)**(-2) - e*(shift)**(-2) + constant

def func_bandwidth():
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