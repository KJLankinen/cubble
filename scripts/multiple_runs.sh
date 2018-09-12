#!/bin/bash

#SBATCH --mem=1G
#SBATCH --time=04:00:00						## wallclock time hh::mm::ss
#SBATCH --gres=gpu:1 --constraint='pascal'			## P100
#SBATCH --mail-user=juhana.lankinen@aalto.fi --mail-type=ALL

module purge
module load goolfc/triton-2017a

cd $WRKDIR/cuda_bubble/
srun make clean
srun make final
sbatch scripts/array_run_dont_use_directly.sh
