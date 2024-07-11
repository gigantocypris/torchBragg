"""
Usage:
libtbx.python $MODULES/torchBragg/tests/tst_convert_coeff.py
"""
import torch
from torchBragg.kramers_kronig.convert_fdp_helper import func, convert_coeff, func_reformatted
torch.set_default_dtype(torch.float64) # important to prevent numerical errors

shift = 1
constant = 2
x = torch.tensor([6500.])
power = torch.tensor([-2,-1,0,1,2,3], dtype = torch.float)[None]
coeff = torch.arange(1.,6.)[:,None]
fdp_0 = func(x, shift, constant, *coeff)
coeff_new = convert_coeff(shift, constant, *coeff)
fdp_1 = func_reformatted(x[:,None], power, coeff_new)
assert torch.allclose(fdp_0, fdp_1, rtol=1e-05, atol=1e-08, equal_nan=False)

shift = 6500.
constant = 2.5358
x = torch.tensor([6500.])
power = torch.tensor([-2,-1,0,1,2,3], dtype = torch.float)[None]
coeff = torch.tensor([[-0.0010], [ 0.0060], [ 0.2629], [ 0.0000], [ 0.0000]], dtype=torch.float64)
fdp_0 = func(x, shift, constant, *coeff)
coeff_new = convert_coeff(shift, constant, *coeff)
fdp_1 = func_reformatted(x[:,None], power, coeff_new)
assert torch.allclose(fdp_0, fdp_1, rtol=1e-05, atol=1e-08, equal_nan=False)