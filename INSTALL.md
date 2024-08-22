# Installation Instructions

These instructions are for installing `torchBragg` and the Computational Crystallography Toolbox `cctbx` on the Perlmutter supercomputer at [NERSC](https://www.nersc.gov/).

Start by opening a terminal on a login node on Perlmutter and run the following:
```
export USERNAME={your_nersc_username}
export PROJECT_ID={mXXX}
module load PrgEnv-gnu cpe-cuda cudatoolkit
mkdir -d /global/cfs/cdirs/${PROJECT_ID}/users/vidyagan/cctbx_install
git clone https://github.com/JBlaschke/alcc-recipes.git alcc-recipes-torchBragg
cd alcc-recipes-torchBragg/cctbx/
```

Open `setup_perlmutter.sh`:
```
vi setup_perlmutter.sh
```

Apply the following patch to dissociate the existing version of the `libssh` library:
```
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
```

The following takes about an hour to complete, recommended to run on a NoMachine instance:
```
./setup_perlmutter.sh > >(tee -a ../../alcc-recipes-torchBragg.log) 2> >(tee -a ../../alcc-recipes-torchBragg.err >&2)
```

Create a file in `$HOME` to source the environment:
``` 
cd
vi env_torchBragg
```

Copy in the following:
```
export USERNAME={your_nersc_username}
export PROJECT_ID={mXXX}
export CFSW=$CFS/${PROJECT_ID}/users/${USERNAME}/cctbx_install
export WORK=$CFSW/evaluate
cd $WORK
source $CFSW/alcc-recipes-torchBragg/cctbx/utilities.sh
source $CFSW/alcc-recipes-torchBragg/cctbx/opt/site/nersc_perlmutter.sh
module load evp-patch
load-sysenv
activate

export MODULES=$CFSW/alcc-recipes-torchBragg/cctbx/modules
export BUILD=$CFSW/alcc-recipes-torchBragg/cctbx/build

export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export KOKKOS_DEVICES="OpenMP;Cuda"
export KOKKOS_ARCH="Ampere80"
export CUDA_LAUNCH_BLOCKING=1
export SIT_DATA=${OVERWRITE_SIT_DATA:-$NERSC_SIT_DATA}:$SIT_DATA
export SIT_PSDM_DATA=${OVERWRITE_SIT_PSDM_DATA:-$NERSC_SIT_PSDM_DATA}
export MPI4PY_RC_RECV_MPROBE='False'
export SIT_ROOT=/reg/g/psdm
```

Source your new file:
```
source ~/env_torchBragg
cd $MODULES
```

Clone the following repos, including `torchBragg`:
```
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
```

Run the following:
```
libtbx.configure LS49 ls49_big_data uc_metrics lunus sim_erice xfel_regression
cd $BUILD
mk-cctbx cuda build # not make!!

cd $MODULES
libtbx.configure LS49 ls49_big_data uc_metrics lunus sim_erice xfel_regression
libtbx.refresh

libtbx.python -m pip install natsort
mkdir -d $WORK/output_torchBragg
```

Test your PyTorch install:
```
libtbx.python
import torch
torch.cuda.is_available()
```


As of June 2024 the followind branches are needed:
```
cd $MODULES/cctbx_project
git pull
git checkout memory_policy


cd $MODULES/cctbx_project
git pull
git checkout dsp_oldstriping
```

These packages may need to be pulled as well to update an old install:
```
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
```

Anytime C++ code is re-pulled, run the following:
```
cd $MODULES
libtbx.configure LS49 ls49_big_data uc_metrics lunus sim_erice xfel_regression
cd $BUILD
mk-cctbx cuda build # not make!!

cd $MODULES
libtbx.configure LS49 ls49_big_data uc_metrics lunus sim_erice xfel_regression
libtbx.refresh
```

To run some code, the following patch is needed in `cctbx_project`:
```
--- a/xfel/merging/command_line/merge.py
+++ b/xfel/merging/command_line/merge.py
@@ -44,6 +44,7 @@ class Script(object):
   def __init__(self):
     self.mpi_helper = mpi_helper()
     self.mpi_logger = mpi_logger()
+    self.common_store = dict(foo="hello") # always volatile, no serialization, no particular dict keys guaranteed
 
   def __del__(self):
     self.mpi_helper.finalize()
@@ -163,6 +164,7 @@ class Script(object):
     # Perform phil validation up front
     for worker in workers:
       worker.validate()
+      worker.__dict__["common_store"] = self.common_store
     self.mpi_logger.log_step_time("CREATE_WORKERS", True)
 
     # Do the work
```