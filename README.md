# torchBragg

Structure factor refinement integrating Careless and nanoBragg

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
> sbatch $MODULES/exafel_project/kpp-sim/sim_ferredoxin_high_remote.sh
```

## Notes

Expanded path to $MODULES: /global/cfs/cdirs/m3562/users/vidyagan/p20231/alcc-recipes2/cctbx/modules

