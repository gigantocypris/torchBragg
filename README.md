# torchBragg

Differentiable version of nanoBragg, used for SPREAD optimization with a Kramer's-Kronig constraint.

## Installation

Follow the [installation instructions](INSTALL.md) to setup CCTBX with PyTorch integration.

## Optimize the anomalous scattering factors

login to Perlmutter

Open folder on sidebar in VSCode:
/global/cfs/cdirs/m3562/users/vidyagan/cctbx_install/alcc-recipes-2/cctbx/modules/torchBragg

source ~/env_feb_2024 
cd $WORK/output_torchBragg 
libtbx.python $MODULES/torchBragg/kramers_kronig/create_fp_fdp_dat_file.py # only run on the first time 

libtbx.python $MODULES/torchBragg/kramers_kronig/convert_fdp.py --prefix Mn2O3_spliced # only need to run on the first time, run for "Mn2O3_spliced" 
libtbx.python $MODULES/torchBragg/kramers_kronig/convert_fdp.py --prefix MnO2_spliced # only run on the first time, run for "MnO2_spliced" 

For visualizations, open a second VSCode window on Perlmutter and open folder:
/global/cfs/cdirs/m3562/users/vidyagan/cctbx_install/evaluate/output_torchBragg

Back to original VSCode window:
salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 1 --account=m3562_g

. $MODULES/torchBragg/tests/tst_torchBragg_psii_script.sh

. $MODULES/torchBragg/scripts/anomalous_optimizer_script.sh 


Question to answer:
Does implementing the Kramer's Kronig restraint improve the SPREAD optimization?
Compare to the previous constraint (current state-of-art) in Sauter 2020.

# Running notes

To start up in a new Perlmutter terminal:
```
source ~/env_torchBragg
cd $WORK/output_torchBragg
```

Created $MODULES/torchBragg/SPREAD_integration/11sfactors.sh (cut down .expt and .refl to just 1 file each)

salloc --nodes 4 --qos interactive --time 01:00:00 --constraint gpu --account=m3562_g --ntasks-per-gpu 1

cd $WORK/output_torchBragg
. $MODULES/torchBragg/SPREAD_integration/11sfactors.sh


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

nanoBragg_nks.cpp or nanoBragg.cpp or KOKKOS nanoBragg: what are differences, what should I use?

## Known Issues

- Interpolation?
- The polarization factor is computed in the innermost loop. However, it is applied in the outermost loop
