#!/bin/bash

#SBATCH --mem=1G
#SBATCH --time=04:00:00						## wallclock time hh:mm:ss
#SBATCH --gres=gpu:1 --constraint='pascal'		## use K80 or P100
#SBATCH --mail-user=juhana.lankinen@aalto.fi --mail-type=ALL

module purge
module load gcc/5.5.0
module load cuda/9.2.88
module load vtk/8.0.1-opengl2-osmesa-python2

srun make clean
srun make final
##srun --gres=gpu:1 nvprof --profile-from-start off --print-gpu-trace bin/cubble input_parameters.json output_parameters.json
srun --gres=gpu:1 bin/cubble input_parameters.json output_parameters.json
