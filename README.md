# torchBragg

Structure factor refinement integrating Careless and nanoBragg

## How to run a forward simulation with nanoBragg

Use the [exafel_project](https://github.com/ExaFEL/exafel_project) repository.
```
git clone https://github.com/ExaFEL/exafel_project.git
git checkout experimental_high_remote
```

Login to Perlmutter, source your environment, and create a simulated image:
```
source ~\env_p20231_2
cd $MODULES
cd ../
. activate.sh
mkdir $WORK/output_torchBragg
cd $WORK/output_torchBragg
> sbatch $MODULES/exafel_project/kpp-sim/sim_ferredoxin_high_remote.sh
```
