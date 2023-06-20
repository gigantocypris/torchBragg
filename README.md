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

## Port to PyTorch

References:

cctbx_project/simtbx/nanoBragg/nanoBragg.cpp
cctbx_project/simtbx/nanoBragg/tst_nanoBragg_minimal.py

> cd $WORK/output_torchBragg
> libtbx.python $MODULES/cctbx_project/simtbx/nanoBragg/tst_nanoBragg_minimal.py

