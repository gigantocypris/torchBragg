# Usage:
# source my_env
# cd $WORKING_DIR

# # Only have to run the following once in working directory:
# libtbx.python $MODULES/torchBragg/kramers_kronig/create_fp_fdp_dat_file.py 
# libtbx.python $MODULES/torchBragg/kramers_kronig/convert_fdp.py --prefix Mn2O3_spliced
# libtbx.python $MODULES/torchBragg/kramers_kronig/convert_fdp.py --prefix MnO2_spliced

# salloc --qos shared_interactive --time 01:00:00 --constraint gpu --gpus 1 --account=m4734_g
# salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 4 --account=m4734_g
# . $MODULES/torchBragg/scripts/anomalous_optimizer_script.sh

export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1
export DEVICES_PER_NODE=1

export N_START=0
export LOG_BY_RANK=0 # Set to 1 to use Aaron's rank logger
export RANK_PROFILE=0 # 0 or 1 Use cProfiler, default 1
export N_SIM=1 # total number of images to simulate
export ADD_BACKGROUND_ALGORITHM=cuda
export MOS_DOM=6 # 26

export CCTBX_NO_UUID=1
export DIFFBRAGG_USE_KOKKOS=1
export CUDA_LAUNCH_BLOCKING=1
export NUMEXPR_MAX_THREADS=128
export SLURM_CPU_BIND=cores # critical to force ranks onto different cores. verify with ps -o psr <pid>
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export SIT_PSDM_DATA=/global/cfs/cdirs/lcls/psdm-sauter
export MPI4PY_RC_RECV_MPROBE='False'

echo "
noise=True
psf=False
attenuation=True
context=kokkos_gpu
absorption=spread
oversample=1
crystal {
  structure=PSII
  #structure=pdb
  #pdb.code=None
  #pdb.source=file
  #pdb.file=${MODULES}/ls49_big_data/7RF1_refine_030_Aa_refine_032_refine_034.pdb
  length_um=2000000
}
beam {
  mean_energy=6550.
  #mean_energy=9500.
}
spectrum {
  nchannels=5
  channel_width=10.0
}
detector {
  tiles=single
  #reference=${MODULES}/exafel_project/kpp-sim/t000_rg002_chunk000_reintegrated_000000.expt
  offset_mm=0 #1.4
}
output {
  format=h5
}
" > optimizer_params.phil

echo "jobstart $(date)";pwd
libtbx.python $MODULES/torchBragg/kramers_kronig/anomalous_optimizer.py optimizer_params.phil
echo "jobend $(date)";pwd