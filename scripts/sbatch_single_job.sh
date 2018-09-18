#!/bin/bash

#SBATCH --mem=1G
#SBATCH --time=04:00:00						## wallclock time hh:mm:ss
#SBATCH --gres=gpu:1 --constraint='pascal'		## use K80 or P100
#SBATCH --mail-user=juhana.lankinen@aalto.fi --mail-type=ALL

module purge
module load goolfc/triton-2017a					## toolchain

srun make clean
srun make final
##srun --gres=gpu:1 nvprof --profile-from-start off --print-gpu-trace bin/cubble input_data.json save.json
srun --gres=gpu:1 bin/cubble input_data.json save.json
