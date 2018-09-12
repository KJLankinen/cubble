#!/bin/bash

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ====================================================================================================
# DON'T RUN this script directly with bash/sbatch. This is called by another script,
# which first compiles the binaries, so each run uses the same binary and they don't all have to do it.
# ====================================================================================================
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

module purge
module load goolfc/triton-2017a

cd $WRKDIR/cuda_bubble/multiple_runs_data/$SLURM_ARRAY_ID_TASK/
cat data/input_data.json
echo "srun --gres=gpu:1 $WRKDIR/cuda_bubble/bin/cubble input_data.json save.json"
