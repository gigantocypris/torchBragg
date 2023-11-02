# October 18, 2023

Setup:
- Login to Perlmutter
- Run the following code:
```
source ~/env_ecp
cd $MODULES
cd ../
. activate.sh
cd $WORK/output_torchBragg
```
- Note: use NoMachine for visualizations (e.g. for dials.image_viewer)

What we need to do:
1. Create simulated images
2. Run dials.stills_process and cctbx.xfel.merge on the simulated images, yielding orientation, unit cell, structure factors, etc
3. Run a nanoBragg simulation with the parameters found in dials.stills_process and cctbx.xfel.merge and try to replicate the simulated image (reference step). All the differentiable operations will need to be re-written in PyTorch
4. Using all the shoeboxes are tokens (integrated shoeboxes or indexed shoeboxes? Note the indexed shoeboxes are not the same size), process through the encoder of a transformer architecture and get to the latent space. Include a positional encoding denoting the coordinates on the image as well as the miller indices/reciprocal lattice space coordinates.
5. Sample the latent space, and process through a decoder (need to think what architecture will be like) to get the orientation and unit cell (these could be deltas on the DIALS values)
6. Structure factors will have a Wilson prior (or no prior, just maximizing the likelihood), and the lattice will not be applied
7. Sample the structure factors, apply the lattice due to the sampled unit cell
8. With all the sampled values, run in the new PyTorch add_spots function. Assume detector metrology, beam, other crystal parameters known for the initial iteration
9. Optimize so that the VAE loss is minimized at all the original shoebox positions


Steps 1-2 are complete for thermolysin. See the results detailed in `/global/cfs/cdirs/m3562/users/vidyagan/p20231/alcc-recipes-spread/cctbx/modules/exafel_project/kpp-sim/thermolysin/README.md`

Output from the thermolysin processing is in:
$SCRATCH/thermolysin

Evaluation of the results are in:
$WORK/thermolysin

Simulated images are in:
$SCRATCH/thermolysin/13719137

To view:
dials.image_viewer image_rank_01517.h5

Indexed/integrated images:
$WORK/thermolysin/14199866

Viewing indexed image:
dials.image_viewer idx-0155_indexed.refl idx-0155_refined.expt

Viewing integrated image:
dials.image_viewer idx-0155_integrated.refl idx-0155_integrated.expt

Merging results are in:
14297594/out (why does merging output .refl files? aggregates data about all the spots. pixel values appear to be missing)



To get data from MTZ file, look at: /global/cfs/cdirs/m3562/users/vidyagan/p20231/alcc-recipes-spread/cctbx/modules/exafel_project/kpp-sim/thermolysin/evaluate_100k.py

TODO: Look into diffBragg to figure out how to simulate the shoeboxes

TODO: Replicate indexed or integrated image with nanoBragg?

Useful notes:
'aaron_notes_creating_worker_image_pixels_nov_18_2022'

Useful commands:
Counting certain number of a type of file in current directory
find . -type f -name "*.refl" | wc -l

viewing just the reflections:
dials.reflection_viewer


Create the file: 14310365_integ_exp_ref.txt --> this is the "exp_ref_spec_file"
This file lists all the refined/indexed expt/refl pairs, with a number indexing the single still shot
Need to iterate through this file and simulate each of the constituent images and compare to original image simulation


Extracting information from an *.expt file:
cd /pscratch/sd/v/vidyagan/thermolysin/14199866
libtbx.python
>> from dxtbx.model import ExperimentList
>> ab=ExperimentList.from_file('idx-0103_refined.expt')
>> ab[0]
>> ab[0].crystal
>> ab[0].detector
>> ab[0].beam
>> ab[0].crystal.get_unit_cell()

Notes from diffBragg code:

## hopper (stage 1):

using indexed and refined shoeboxes, saves the data and values to simulate the data in the pkl file

### Forward Simulation

/global/cfs/cdirs/m3562/users/vidyagan/p20231/alcc-recipes-spread/cctbx/modules/cctbx_project/simtbx/diffBragg/utils.py
Line 794: simulator_from_expt_and_params

/global/cfs/cdirs/m3562/users/vidyagan/p20231/alcc-recipes-spread/cctbx/modules/cctbx_project/simtbx/diffBragg/hopper_utils.py
Line 973: Minimize
Line 1679: TargetFunc
Line 1804: model_pix = model_bragg + background
Line 1790: model_bragg, Jac = model(x, mod, SIM, compute_grad=_compute_grad)
SIM.D.add_diffBragg_spots, then get raw pixels from SIM


### Gathering shoeboxes

/global/cfs/cdirs/m3562/users/vidyagan/p20231/alcc-recipes-spread/cctbx/modules/cctbx_project/simtbx/diffBragg/hopper_utils.py

line 361 GatherFromReflectionTable

### output

cd /pscratch/sd/v/vidyagan/thermolysin/14310798/stage1/pandas

.pkl files are output:
```
libtbx.python
import pandas
rank = 0
df = pandas.read_pickle("hopper_results_rank" + str(rank) + ".pkl")
print(df.keys()) # print column names
print(df.iloc[0]) # print row 0
print(df.iloc[0]['lam0']) # print row 0, column 'lam0'
```
I don't think the pixels of the predicted shoeboxes are saved, only refined quantities (e.g. rotation) that will allow simulation of spot
Note: each row of the pandas tables probably corresponds to a single image

Column names:
['spot_scales', 'Amats', 'ncells', 'spot_scales_init', 'ncells_init',
       'eta_abc', 'detz_shift_mm', 'ncells_def', 'diffuse_gamma',
       'diffuse_sigma', 'fp_fdp_shift', 'use_diffuse_models',
       'gamma_miller_units', 'eta', 'rotX', 'rotY', 'rotZ', 'a', 'b', 'c',
       'al', 'be', 'ga', 'a_init', 'b_init', 'c_init', 'al_init', 'lam0',
       'lam1', 'be_init', 'ga_init', 'spectrum_filename', 'spectrum_stride',
       'total_flux', 'beamsize_mm', 'exp_name', 'opt_exp_name',
       'spectrum_from_imageset', 'oversample', 'stage1_refls',
       'stage1_output_img', 'exp_idx', 'sigz', 'niter', 'phi_deg', 'osc_deg']

## Integrate and predict step
simtbx/command_line/integrate.py

outputs expts and refls
view output:
cd /pscratch/sd/v/vidyagan/thermolysin/14314404/predict/expts_and_refls
dials.image_viewer rank250_preds.refl rank250_preds.expt
dials.reflection_viewer rank250_preds.refl

I think this step outputs the predicted refls and expts from simulation values, and draws the integrated shoeboxes

STOPPED HERE

## diffBragg stage 2


# Transformer idea

Shoeboxes are tokens
Process all shoeboxes with transformer architecture to create a representation. The representation is processed to determine the orientation and unit cell distributions (Structure factor matrix is optimized directly). Orientation, unit cell, structure factor matrix processed by the forward model to create the simulated image.


# Reference

Count number of files with extension .ext in current folder:
find . -type f -name "*.ext" | wc -l

All all files in current folder:
find . -type f | wc -l

## October 25, 2023

Compare how KOKKOS works against CPU: /global/cfs/cdirs/m3562/users/vidyagan/p20231/alcc-recipes-spread/cctbx/modules/cctbx_project/simtbx/kokkos/tst_kokkos_lib.py

Where is the unit cell in LY99_batch.py?


Single image simulation:

export SCRATCH_FOLDER=$WORK/output_torchBragg
mkdir -p $SCRATCH_FOLDER; cd $SCRATCH_FOLDER

export CCTBX_DEVICE_PER_NODE=1
export N_START=0
export LOG_BY_RANK=1 # Use Aaron's rank logger
export RANK_PROFILE=0 # 0 or 1 Use cProfiler, default 1
export N_SIM=1 # total number of images to simulate
export ADD_BACKGROUND_ALGORITHM=cuda
export DEVICES_PER_NODE=4
export MOS_DOM=25

export CCTBX_NO_UUID=1
export DIFFBRAGG_USE_KOKKOS=1
export CUDA_LAUNCH_BLOCKING=1
export NUMEXPR_MAX_THREADS=128
export SLURM_CPU_BIND=cores # critical to force ranks onto different cores. verify with ps -o psr <pid>
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export SIT_PSDM_DATA=/global/cfs/cdirs/lcls/psdm-sauter
export CCTBX_GPUS_PER_NODE=1
export XFEL_CUSTOM_WORKER_PATH=$MODULES/psii_spread/merging/application # User must export $MODULES path

# determine oversample with simple difference script IN PROCESS
# check that both sim and expt have gain==1
# define ncells as 10x10x10 ???  not sure how ncells is now gotten
# store constant background for all images
# xtal size mm 0.00015
# mos domains 50 (my default 25)
# mos spread deg 0.01 (my default 0.05)
# --masterscale 1150 --sad --bs7real --masterscalejitter 115

echo "
noise=True
psf=False
attenuation=True
context=kokkos_gpu
absorption=high_remote
oversample=1
beam {
  mean_energy=9500.
}
spectrum {
  nchannels=100
  channel_width=1.0
}
crystal {
  # Perlmutter OK-download in job from PDB
  # structure=pdb
  # pdb.code=4tnl # thermolysin
  # Frontier OK-take PDB file from github
  structure=pdb
  pdb.code=None
  pdb.source=file
  pdb.file=${MODULES}/exafel_project/kpp-sim/thermolysin/4tnl.pdb
  length_um=0.5 # increase crystal path length
}
detector {
  tiles=multipanel
  reference=$MODULES/exafel_project/kpp-sim/t000_rg002_chunk000_reintegrated_000000.expt
  offset_mm=80.0 # desired 1.8 somewhere between inscribed and circumscribed.
}
output {
  format=h5
  ground_truth=${SCRATCH_FOLDER}/ground_truth.mtz
}
" > trial.phil

export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1

libtbx.python $MODULES/exafel_project/kpp_utils/LY99_batch.py trial.phil

# October 26, 2023

Get unit cell, line 92 in /global/cfs/cdirs/m3562/users/vidyagan/p20231/alcc-recipes-spread/cctbx/modules/LS49/sim/util_fmodel.py
xray_structure.unit_cell()

Single unit cell for initial prototype:
1. Create PyTorch equivalent of nanoBragg C++ object, torchBragg
2. Simulate with nanoBragg and simulate with torchBragg, compare --> test
3. Simulate a dataset with nanoBragg, index, integrate, and merge (merged SF can be used as warm start), use integrated shoeboxes (or make spotfinder/indexed shoeboxes equal size)
4. Vision transformer (may need to add null padding for variable numbers of shoeboxes) to get latent space for the orientation, decode and sample to get orientation
5. Run forward solution to get shoebox mean and variances, transform these by a neural network to get corrected mean and variance

Unit test:
libtbx.python $MODULES/cctbx_project/simtbx/nanoBragg/tst_nanoBragg_minimal.py >> output_tst_nanoBragg_minimal.txt
libtbx.python $MODULES/cctbx_project/simtbx/nanoBragg/tst_nanoBragg_basic.py >> output_tst_nanoBragg_basic.txt

For KOKKOS, looks like the work happends here: /global/cfs/cdirs/m3562/users/vidyagan/p20231/alcc-recipes-spread/cctbx/modules/cctbx_project/simtbx/kokkos/simulation_kernels.h

Run test:
> cd $WORK/output_torchBragg
> libtbx.python $MODULES/torchBragg/tst_torchBragg_minimal.py

# October 27, 2023

/global/cfs/cdirs/m3562/users/vidyagan/p20231/alcc-recipes-spread/cctbx/modules/cctbx_project/simtbx/nanoBragg/nanoBragg.cpp

Line 2462: add_nanoBragg_spots

# October 30, 2023

Bulk solvent: https://journals.iucr.org/d/issues/2013/04/00/dz5273/index.html

# October 31, 2023

> cd $WORK/output_torchBragg
> libtbx.python $MODULES/cctbx_project/simtbx/nanoBragg/tst_nanoBragg_basic.py


Noise:
Add flicker noise (additive Gaussian)
Add Poisson noise
Add calibration noise (additive Gaussian that is the same for all shots in the experiment)

Implement PSF here

Add read out noise:
  Convert photon signal to read out units (photon signal is floatimage[i]): adu = floatimage[i]*quantum_gain + adc_offset
  Additive Gaussian readout noise, readout_noise is in adu units: adu += readout_noise * image_deviates.gaussdev( &seed );

Notes for torchBragg: do an additive sum of normal distributions, including approximating the Poisson distribution as a normal.

In add_noise, the only approximation is the addition of Poisson noise. First of all, we take the Gaussian distribution of the Poisson distribution. Secondly, to add Poisson noise, nanoBragg first adds flicker noise, then samples the distribution. The sampled value becames the Poisson parameter (the mean and variance of the approximate Gaussian distribution). In the torchBragg code, we add the flicker noise deterministically and then apply the Poisson approximate distribution. This means that we take a single Monte Carlo sample of the flicker noise to approximate the distribution.


torchBragg basic test:
> cd $WORK/output_torchBragg
> libtbx.python $MODULES/torchBragg/tst_torchBragg_basic.py