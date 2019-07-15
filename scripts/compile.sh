#!/bin/bash
#SBATCH --job-name=cubble_compile
#SBATCH --mem=100M
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1  
#SBATCH --constraint=pascal
module load cuda/10.0.130 gcc/6.3.0
mkdir -p /tmp/$SLURM_JOB_ID $WRKDIR/cubble/data/compiled
srun make -C $WRKDIR/cubble/final BIN_PATH=/tmp/$SLURM_JOB_ID
cp /tmp/$SLURM_JOB_ID/cubble $WRKDIR/cubble/data/compiled/. 
