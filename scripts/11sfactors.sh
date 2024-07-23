# Optimization of fdp and fp with CCTBX, not torchBragg
# Usage:
# salloc --nodes 4 --qos interactive --time 01:00:00 --constraint gpu --account=m3562_g --ntasks-per-gpu 1
# cd $WORK/output_torchBragg
# . $MODULES/torchBragg/scripts/11sfactors.sh

mkdir -p $SLURM_JOB_ID; cd $SLURM_JOB_ID

export CCTBX_NO_UUID=1
export DIFFBRAGG_USE_CUDA=1
export CUDA_LAUNCH_BLOCKING=1
export NUMEXPR_MAX_THREADS=128
export SLURM_CPU_BIND=cores # critical to force ranks onto different cores. verify with ps -o psr <pid>
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export SIT_PSDM_DATA=/global/cfs/cdirs/lcls/psdm-sauter
export SIT_ROOT=/reg/g/psdm
export SIT_DATA=/global/common/software/lcls/psdm/data
export CCTBX_GPUS_PER_NODE=1
export XFEL_CUSTOM_WORKER_PATH=$MODULES/psii_spread/merging/application # User must export $MODULES path
export WERK=/global/cfs/cdirs/m3562/users/dtchon/p20231/common/ensemble1/SPREAD/SIM/9exa3a
export DWORK=/global/cfs/cdirs/m3562/users/dtchon/p20231/common/ensemble1/SPREAD/SIM/10sfactors/22948326

echo "
dispatch.step_list = input arrange annulus
input.path=$WERK/22374020_e0.05Nabc16/out_P212121
input.experiments_suffix=000000.expt  # *1.expt # testing on 10% of data
input.reflections_suffix=000000.refl  # *1.refl # testing on 10% of data
input.keep_imagesets=True
input.read_image_headers=False
input.persistent_refl_cols=shoebox
input.persistent_refl_cols=bbox
input.persistent_refl_cols=xyzcal.px
input.persistent_refl_cols=xyzcal.mm
input.persistent_refl_cols=xyzobs.px.value
input.persistent_refl_cols=xyzobs.mm.value
input.persistent_refl_cols=xyzobs.mm.variance
input.persistent_refl_cols=delpsical.rad
input.persistent_refl_cols=panel
input.parallel_file_load.method=uniform
input.parallel_file_load.balance=global1
input.parallel_file_load.balance_mpi_alltoall_slices = 50
arrange.sort.by=imageset_path
scaling.model=$MODULES/ls49_big_data/7RF1_refine_030_Aa_refine_032_refine_034.pdb
scaling.unit_cell=117.463  222.609  309.511  90.00  90.00  90.00
scaling.space_group=P212121
scaling.resolution_scalar=0.96
scaling.pdb.k_sol=0.435
# filter.unit_cell.cluster.covariance.file=$WERK/reference/covariance_run145_cells.pickle
# filter.unit_cell.cluster.covariance.component=0
# filter.unit_cell.cluster.covariance.mahalanobis=2.0
merging.d_max=None
merging.d_min=3.2
statistics.annulus.d_max=5.0  # consider annulus thinner than original 5.0-3.2 for tests?
statistics.annulus.d_min=3.2  # consider annulus thinner than original 5.0-3.2 for tests?
spread_roi.enable=True
# spread_roi.strong=1.0 # only use for initial annulus definition, not subsequent
output.log_level=0 # 0 = stdout stderr, 1 = terminal
output.output_dir=out
output.prefix=trial8_scenario3A
output.save_experiments_and_reflections=True
exafel.scenario=S1
exafel.static_fcalcs.path=$DWORK/psii_static_fcalcs.pickle
exafel.static_fcalcs.whole_path=$DWORK/psii_miller_array.pickle
exafel.static_fcalcs.action=read
exafel.trusted_mask=$CFS/m3562/users/dtchon/p20231/common/ensemble1/SPREAD/SIM/JungFrau16_void.mask
exafel.shoebox_border=0
exafel.context=kokkos_gpu
exafel.model.plot=False
exafel.model.mosaic_spread.value=0.05
exafel.model.Nabc.value=16,16,16
exafel.debug.lastfiles=False # write out *.h5, *.mask for each image
exafel.debug.verbose=False
exafel.debug.finite_diff=-1
exafel.debug.eps=1.e-8
exafel.debug.format_offset=0
exafel.debug.energy_offset_eV=0
exafel.debug.energy_stride_eV=4.00
exafel.skin=False # whether to use diffBragg
exafel{
  refpar{
    label = *background *G
    background {
      algorithm=rossmann_2d_linear
      scope=spot
      slice_init=border
      slice=all
    }
    G {
      scope=lattice
      reparameterize=bound
    }
  }
}
exafel.metal=PSII  # if memory is not an issue, consider PSII4
sauter20.LLG_evaluator.enable_plot=True
sauter20.LLG_evaluator.title=tell
sauter20.LLG_evaluator.restraints.fp.mean=None
sauter20.LLG_evaluator.restraints.fp.sigma=4.0
sauter20.LLG_evaluator.restraints.fdp.mean=None
sauter20.LLG_evaluator.restraints.fdp.sigma=8.0
sauter20.LLG_evaluator.restraints.kramers_kronig.use=False
sauter20.LLG_evaluator.restraints.kramers_kronig.pad=100
sauter20.LLG_evaluator.restraints.kramers_kronig.trim=0
sauter20.LLG_evaluator.restraints.kramers_kronig.weighting_factor=1000.0
sauter20.LLG_evaluator.max_calls=2 # 30
" > refine.phil
echo "jobstart $(date)";pwd
srun --ntasks 4 --cpus-per-task 128 --ntasks-per-gpu=1 cctbx.xfel.merge refine.phil
echo "jobend $(date)";pwd