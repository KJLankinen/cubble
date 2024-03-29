#!/bin/bash

#SBATCH --job-name=cubble
#SBATCH --account=project_2002078
#SBATCH --mail-type=ALL

##SBATCH --mem-per-cpu=16G              ## How much memory per CPU
#SBATCH --partition=gpu                 ## Use the gpu partition
#SBATCH --time=04:00:00                 ## Wall clock time, 4h
#SBATCH --gres=gpu:v100:1               ## One volta
##SBATCH --gres=nvme:10                 ## 100 GB at $LOCAL_SCRATCH

module load gcc/11.3.0 cuda/11.7.0 cmake/3.23.1

## Build
srun cp -r $HOME/Code/cubble /scratch/project_2002078/$USER/
srun cd /scratch/project_2002078/$USER/cubble
srun scripts/build.sh Release .

## Run
srun cubble/v0.1.0.0/Release/bin/cubble-cli input_parameters.json
