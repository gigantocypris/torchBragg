## These are February 2024 install instructions. See the end for modifications in June 2024

module load PrgEnv-gnu cpe-cuda cudatoolkit
cd /global/cfs/cdirs/m3562/users/vidyagan/cctbx_install
git clone https://github.com/JBlaschke/alcc-recipes.git alcc-recipes-2
cd alcc-recipes-2/cctbx/

vi setup_perlmutter.sh

Apply the following patch to dissociate the existing version of the libssh library:
a/cctbx/setup_perlmutter.sh b/cctbx/setup_perlmutter.sh
index 3600cf8..56b8d44 100755
--- a/cctbx/setup_perlmutter.sh
+++ b/cctbx/setup_perlmutter.sh@@ -31,6 +31,7 @@ if fix-sysversions
 then
     return 1
 fi
+rm ./opt/mamba/envs/psana_env/lib/libssh.so.4
 mk-cctbx cuda build hot update
 patch-dispatcher nersc

Following takes about an hour to complete:

./setup_perlmutter.sh > >(tee -a ../../alcc-recipes-2.log) 2> >(tee -a ../../alcc-recipes-2.err >&2)

STOPPED HERE (11:44 am, the above command is running on NoMachine)

cd
vi env_feb_2024

###
export CFSW=$CFS/m3562/users/vidyagan/cctbx_install
export WORK=$CFSW/evaluate
cd $WORK
source $CFSW/alcc-recipes-2/cctbx/utilities.sh
source $CFSW/alcc-recipes-2/cctbx/opt/site/nersc_perlmutter.sh
module load evp-patch
load-sysenv
activate

export MODULES=$CFSW/alcc-recipes-2/cctbx/modules
export BUILD=$CFSW/alcc-recipes-2/cctbx/build
export SCRATCH=/pscratch/sd/v/vidyagan

export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export KOKKOS_DEVICES="OpenMP;Cuda"
export KOKKOS_ARCH="Ampere80"
export CUDA_LAUNCH_BLOCKING=1
export SIT_DATA=${OVERWRITE_SIT_DATA:-$NERSC_SIT_DATA}:$SIT_DATA
export SIT_PSDM_DATA=${OVERWRITE_SIT_PSDM_DATA:-$NERSC_SIT_PSDM_DATA}
export MPI4PY_RC_RECV_MPROBE='False'
export SIT_ROOT=/reg/g/psdm

###

source ~/env_feb_2024
cd $MODULES

git clone https://github.com/nksauter/LS49 && \
git clone https://gitlab.com/cctbx/ls49_big_data && \
git clone https://gitlab.com/cctbx/uc_metrics && \
git clone https://github.com/lanl/lunus && \            
git clone https://github.com/dermen/sim_erice && \
git clone https://gitlab.com/cctbx/psii_spread.git && \	
git clone https://gitlab.com/cctbx/xfel_regression.git && \
git clone https://github.com/ExaFEL/exafel_project.git && \
git clone https://github.com/dermen/cxid9114.git && \
git clone https://github.com/gigantocypris/torchBragg.git

libtbx.configure LS49 ls49_big_data uc_metrics lunus sim_erice xfel_regression
cd $BUILD
mk-cctbx cuda build (not make!!)

cd $MODULES
libtbx.configure LS49 ls49_big_data uc_metrics lunus sim_erice xfel_regression
libtbx.refresh

Note: pytorch still not using GPU, but at least back to where I started? Can make a separate optimized script/environment for GPU and PyTorch. Assume that end user won't be using the full CCTBX stack. Can have instructions for setting up the full stack and getting the initial values/matrices and saving, and then separate for PyTorch optimization
UPDATE: PyTorch using GPU after Felix's modification

FYI for starting VSCode:
> echo $MODULES
/global/cfs/cdirs/m3562/users/vidyagan/cctbx_install/alcc-recipes-2/cctbx/modules

#### starting from scratch
source ~/env_feb_2024
mkdir $WORK
mkdir $WORK/output_torchBragg
cd $WORK/output_torchBragg
. $MODULES/torchBragg/tst_anomalous_optimizer_script.sh

libtbx.python
import torch
torch.cuda.is_available()


After install:

libtbx.python -m pip install natsort


## June 2024 modifications

Resources:
Daniel's SPREAD reprocessing journal: https://docs.google.com/document/d/1Cscemlh67JZjyaY82-sLmnv_bjdJOh44JzxtRe-nWN0/edit
See entry on June 13, 2024, page 119

SPREAD code map: https://docs.google.com/presentation/d/1MfcAD4GOzHFgpsnH4xoRgHETgw90ULTO2hfAisVDWNc/edit#slide=id.p

Important code: https://gitlab.com/cctbx/psii_spread/-/blob/main/merging/application/annulus/new_global_fdp_refinery.py#L153

source ~/env_feb_2024 
cd $MODULES/cctbx_project
git pull
git checkout memory_policy


cd $MODULES/cctbx_project
git pull
git checkout dsp_oldstriping

cd $MODULES/exafel_project
git pull

cd $MODULES/LS49
git pull

cd $MODULES/ls49_big_data
git pull

cd $MODULES/psii_spread
git pull

cd $MODULES/uc_metrics
git pull


libtbx.configure LS49 ls49_big_data uc_metrics lunus sim_erice xfel_regression
cd $BUILD
mk-cctbx cuda build (not make!!)

cd $MODULES
libtbx.configure LS49 ls49_big_data uc_metrics lunus sim_erice xfel_regression
libtbx.refresh

Implement this fix: https://gitlab.com/cctbx/psii_spread/-/blob/main/pipeline/simulated/cctbx.diff?ref_type=heads

Reference file: /global/cfs/cdirs/m3562/users/dtchon/p20231/common/ensemble1/SPREAD/SIM/arrange_tests/10sorted.sh

Created $MODULES/torchBragg/SPREAD_integration/11sfactors.sh (cut down .expt and .refl to just 1 file each)

salloc --nodes 4 --qos interactive --time 01:00:00 --constraint gpu --account=m3562_g --ntasks-per-gpu 1

cd $WORK/output_torchBragg
. $MODULES/torchBragg/SPREAD_integration/11sfactors.sh



Setup after install:
Open in VSCode file explorer: /global/cfs/cdirs/m3562/users/vidyagan/cctbx_install/alcc-recipes-2/cctbx/modules
source ~/env_feb_2024 
cd $WORK/output_torchBragg

VSCode to visualize output: /global/cfs/cdirs/m3562/users/vidyagan/cctbx_install/evaluate/output_torchBragg

File in progress:
$MODULES/torchBragg/SPREAD_integration/coeff_to_fp_fdp.py

libtbx.python $MODULES/torchBragg/SPREAD_integration/helper.py
