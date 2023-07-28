# torchBragg

Structure factor refinement integrating Careless and nanoBragg

## Installation

Follow the [installation instructions](https://github.com/gigantocypris/SPREAD) for SPREAD.

## How to run a forward simulation with nanoBragg

Login to Perlmutter, source your environment, and clone this repository.
```
source ~\env_p20231_2
cd $MODULES
cd ../
. activate.sh
cd $MODULES
git clone https://github.com/gigantocypris/torchBragg.git
```

Switch branches on the [exafel_project](https://github.com/ExaFEL/exafel_project) repository.
```
cd $MODULES/exafel_project
git checkout experimental_high_remote
```

Create a simulated image:
```
mkdir $WORK/output_torchBragg
cd $WORK/output_torchBragg
. $MODULES/torchBragg/single_img_sim.sh
```

To view the image:
```
dials.image_viewer image_rank_00000.h5 
```
Change brightness to ~100.

## Running nanoBragg unit tests

```
cd $WORK/output_torchBragg
libtbx.python $MODULES/cctbx_project/simtbx/nanoBragg/tst_nanoBragg_minimal.py
```

Can set a breakpoint in the above unit test and look at the raw_pixels attribute:
```
SIM.raw_pixels.as_numpy_array()
```

Other unit tests:
```
cd $WORK/output_torchBragg
libtbx.python $MODULES/cctbx_project/simtbx/nanoBragg/tst_nanoBragg_mosaic.py
```

## Port to PyTorch

References:

cctbx_project/simtbx/nanoBragg/nanoBragg.cpp --> add_nanoBragg_spots
cctbx_project/simtbx/nanoBragg/tst_nanoBragg_minimal.py

diffBragg tests are in:
simtbx/diffBragg/tests

Get this unit test to work:

> cd $WORK/output_torchBragg
> libtbx.python $MODULES/cctbx_project/simtbx/nanoBragg/tst_nanoBragg_minimal.py

diffBragg unit test:
> cd $WORK/output_torchBragg
> libtbx.python $MODULES/cctbx_project/simtbx/diffBragg/tests/tst_diffBragg_Fcell_deriv.py

Run main.py
> cd $WORK/output_torchBragg
> libtbx.python $MODULES/torchBragg/main.py


## Questions on nanoBragg.cpp

Why is there interpolation starting on line 2779? Isn't the correct thing to do is take all the neighboring HKL spots, loop over these spots, getting the contribution of that spot to the pixel of interest?

Some variables in add_nanoBragg_spots are not accessible in the nanoBragg() object, such as pixel_size:
That is, this code snippet prints `None`.
```
SIM = nanoBragg()
print(SIM.pixel_size)
```

The polarization factor is computed in the innermost loop. However, it is applied in the outermost loop. Why is this?

