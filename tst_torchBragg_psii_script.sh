# . $MODULES/torchBragg/tst_torchBragg_psii_script.sh

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
  nchannels=2
  channel_width=1.0
}
detector {
  tiles=single
  #reference=${MODULES}/exafel_project/kpp-sim/t000_rg002_chunk000_reintegrated_000000.expt
  offset_mm=0 #1.4
}
output {
  format=h5
}
" > trial.phil

echo "jobstart $(date)";pwd
libtbx.python $MODULES/torchBragg/tst_torchBragg_psii.py trial.phil
echo "jobend $(date)";pwd

# echo "jobstart $(date)";pwd
# libtbx.python $MODULES/exafel_project/kpp_utils/LY99_batch.py trial.phil
# echo "jobend $(date)";pwd