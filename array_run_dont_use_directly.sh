#!/bin/bash

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ====================================================================================================
# DON'T RUN this script directly with bash/sbatch. This is called by another script,
# which first compiles the binaries, so each run uses the same binary and they don't all have to do it.
# ====================================================================================================
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#SBATCH --mem=1G
#SBATCH --time=04:00:00						## wallclock time hh::mm::ss
#SBATCH --gres=gpu:1 --constraint='pascal'			## P100
#SBATCH --mail-user=juhana.lankinen@aalto.fi --mail-type=ALL
#SBATCH --array=0-2

module purge
module load goolfc/triton-2017a

RUN_DIR=run_$SLURM_ARRAY_TASK_ID

echo "mkdir $RUN_DIR $RUN_DIR/data"
echo "cp $WRKDIR/cuda_bubble/data/input_data.json $RUN_DIR/data/."
echo "cd $RUN_DIR"
echo "srun --gres=gpu:1 $WRKDIR/cuda_bubble/bin/cubble input_data.json save.json"
