#!/bin/bash
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --constraint='volta'

module load OpenMPI/4.0.1-GCC-8.3.0-2.32
module load cuda/10.1.243

srun /scratch/work/lankinj5/cubble/bin/debug/cubble /scratch/work/lankinj5/cubble/input_parameters.json
