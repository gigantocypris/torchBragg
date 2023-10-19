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
2. Run dials.stills_process and cctbx.xfel.merge on the simulated images, yielding shoeboxes and other parameters
3. Go through the shoeboxes, and run a nanoBragg simulation with the shoebox/image parameters
4. Create a new repo, deepBragg, which is a copy of the CT_NVAE repo
5. Input the shoebox to the NVAE architecture, along with the structure factors, orientation, unit cell, etc. Structure factors are the top hierarchy level (like the ring artifact in the CT problem). For this first test, the structure factors are deterministic, but are a variable that can be optimized. The output of the parameter projection network are deltas to the orientation and unit cell. These go into the nanoBragg forward simulation, and we optimize the likelihood and minimize KL divergence of the latent space.

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
14297594/out (why does merging output .refl files? aggregates data about all the spots)



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