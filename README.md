# torchBragg

Differentiable version of nanoBragg, used for SPREAD optimization with a Kramer's-Kronig constraint.

## Installation

Follow the [installation instructions](INSTALL.md) to setup CCTBX with PyTorch integration.

## Getting Started

Open a Perlmutter terminal and start an interactive session:
```
source ~/env_torchBragg
export PYTHONPATH=
cd $WORK/output_torchBragg
```

## Optimize the Anomalous Scattering Factors

### Modification to `psii_spread`

The forward model in `psii_spread` is not automatically differentiable, except the portion implementing the Kramer's Kronig constraint with PyTorch. To run the optimization of the anomalous scattering factors f' and f":

```
salloc --nodes 4 --qos interactive --time 01:00:00 --constraint gpu --account=${PROJECT_ID}_g --ntasks-per-gpu 1
. $MODULES/torchBragg/SPREAD_integration/11sfactors.sh
```

### Fully Differentiable Model
To optimize the anomalous scattering factors with a automatically differentiable model implemented in PyTorch, first run the following once (only needs to be run the first time using these instructions):
```
libtbx.python $MODULES/torchBragg/kramers_kronig/create_fp_fdp_dat_file.py 
libtbx.python $MODULES/torchBragg/kramers_kronig/convert_fdp.py --prefix Mn2O3_spliced 
libtbx.python $MODULES/torchBragg/kramers_kronig/convert_fdp.py --prefix MnO2_spliced 
```

Start an interactive session and run the script:
```
salloc --qos shared_interactive --time 01:00:00 --constraint gpu --gpus 1 --account=${PROJECT_ID}_g
. $MODULES/torchBragg/scripts/anomalous_optimizer_script.sh 
```

## Unit Tests

```
libtbx.python $MODULES/torchBragg/tests/tst_torchBragg_basic_noise.py
```

```
libtbx.python $MODULES/cctbx_project/simtbx/nanoBragg/tst_nanoBragg_minimal.py
```

```
. $MODULES/torchBragg/tests/tst_torchBragg_psii_script.sh
```

### Tests to implement in torchBragg

From nanoBragg:
```
libtbx.python $MODULES/cctbx_project/simtbx/nanoBragg/tst_nanoBragg_mosaic.py
```

From diffBragg:
```
libtbx.python $MODULES/cctbx_project/simtbx/diffBragg/tests/tst_diffBragg_Fcell_deriv.py
```

## Questions

- Interpolation: Why is there interpolation in nanoBragg? Isn't the correct thing to do is take all the neighboring HKL spots, loop over these spots, getting the contribution of that spot to the pixel of interest?
- Polarization factor: The polarization factor is computed in the innermost loop. However, it is applied in the outermost loop.
