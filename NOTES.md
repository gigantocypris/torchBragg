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