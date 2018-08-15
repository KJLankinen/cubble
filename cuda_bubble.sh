#!/bin/bash

#SBATCH --mem=24G
#SBATCH --time=01:00:00						## wallclock time hh:mm:ss
#SBATCH --gres=gpu:1 --constraint='pascal'			## use P100
#SBATCH --mail-user=juhana.lankinen@aalto.fi --mail-type=ALL

module load goolfc/triton-2017a					## toolchain

srun make clean
srun make
srun --gres=gpu:1 nvprof --profile-from-start off --export-profile timeline.prof bin/cubble data.json save.json
##srun --gres=gpu:1 bin/cubble data.json save.json
