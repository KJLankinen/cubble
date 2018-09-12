#!/bin/bash

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ====================================================================================================
# DON'T RUN this script directly with bash/sbatch. This is called by another script,
# which first compiles the binaries, so each run uses the same binary and they don't all have to do it.
# ====================================================================================================
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

PROJECT_ROOT_DIR="$WRKDIR/cuda_bubble"
MUL_RUN_DIR="$PROJECT_ROOT_DIR/multiple_runs_data"
SELF_DIR="$MUL_RUN_DIR/$SLURM_ARRAY_ID_TASK"
BINARY="$PROJECT_ROOT_DIR/bin/cubble"
INPUT_DATA_FILE="input_data.json"

module purge
module load goolfc/triton-2017a

cd $SELF_DIR
cat "$SELF_DIR/data/INPUT_DATA_FILE"
echo "srun $BINARY $INPUT_DATA_FILE save.json"
