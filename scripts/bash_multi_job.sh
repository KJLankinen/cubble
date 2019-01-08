#!/bin/bash

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# RUN THIS SCRIPT WITH BASH, NOT WITH SBATCH
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


#==============================================================
# This script is supposed to be run on the login shell of triton with bash.
# This creates directories & input data files for each line
# specified in 'multiple_runs_data/multiple_runs_params.json'.
#
# Parameters other than the ones specified in the file are taken from
# the default input data file.
#
# After creating the directories & files, this script submits
# an array of jobs using a temporary script created at the end
# of this file. It's not saved anywhere to avoid the risk of
# accidentally using it for running the program.
#==============================================================


#==============================================================
# Variables
#==============================================================

PROJECT_ROOT_DIR="$WRKDIR/cuda_bubble"
PYTHON_SCRIPT="$PROJECT_ROOT_DIR/scripts/generate_input_jsons.py"
MUL_RUN_DIR="$PROJECT_ROOT_DIR/multiple_runs_data"
PARAMS_FILE="$MUL_RUN_DIR/multiple_runs_params.json"
BINARY="$PROJECT_ROOT_DIR/bin/cubble"
INPUT_DATA="input_data.json"
SAVE_DATA="save.json"


#==============================================================
# Clean up old binary
#==============================================================

cd $PROJECT_ROOT_DIR
make clean


#==============================================================
# Generate folders and input files for each run
#==============================================================

NUM_RUNS = python $PYTHON_SCRIPT $MUL_RUN_DIR $PROJECT_ROOT_DIR"/data/"$INPUT_DATA $PARAMS_FILE
printf "$NUM_RUNS"


#==============================================================
# Create temporary shell scripts for submitting the runs with sbatch.
#
# Build script builds the binary.
# Array script launches an array of jobs, but only after build job
# has finished successfully. Number of jobs on array depends on the
# number of lines in the parameter file, as mentioned above.
#==============================================================

#NEWLINE=$'\n'
#BUILD_SCRIPT="#!/bin/sh"${NEWLINE}"module purge"${NEWLINE}"module load goolfc/triton-2017a"${NEWLINE}"cd $PROJECT_ROOT_DIR"${NEWLINE}"srun make final"${NEWLINE}
#ARRAY_SCRIPT="#!/bin/sh"${NEWLINE}"module purge"${NEWLINE}"module load goolfc/triton-2017a"${NEWLINE}"cd $MUL_RUN_DIR/\$SLURM_ARRAY_TASK_ID"${NEWLINE}"srun $BINARY $INPUT_DATA $SAVE_DATA"${NEWLINE}

#BUILD_JOB_ID=$(printf "$BUILD_SCRIPT" | sbatch --mem=1G --time=04:00:00 --gres=gpu:1 --constraint=pascal --mail-user=juhana.lankinen@aalto.fi --mail-type=ALL | tail -1 | awk -v N=4 '{print $N}')
#printf "$ARRAY_SCRIPT" | sbatch --mem=1G --time=24:00:00 --gres=gpu:1 --constraint=pascal --mail-user=juhana.lankinen@aalto.fi --mail-type=ALL --dependency=aftercorr:"$BUILD_JOB_ID" --array=0-$NUM_RUNS
