# torchBragg 
# Implementation of nanoBragg/diffBragg in PyTorch

## Existing nanoBragg code for porting
- cctbx_project/simtbx/nanoBragg/nanoBragg.h
- cctbx_project/simtbx/nanoBragg/nanoBragg.cpp

- port these methods to PyTorch
    - add_nanoBragg_spots


## Requirements
- All nanoBragg and diffBragg unit tests should achieve the same results with torchBragg
- Diffuse scattering code should be easily integrated
- The code should be parallelizable across an arbitrary number of nodes and GPUs
- Integration with KOKKOS?

## Resources

- PyTorch distributed computing

## Questions
- Where should the code be located?
- Interpolation?
- The polarization factor is computed in the innermost loop. However, it is applied in the outermost loop