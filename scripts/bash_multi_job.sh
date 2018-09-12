#!/bin/bash

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# RUN THIS SCRIPT WITH BASH, NOT WITH SBATCH
# i.e. 'bash script_name.sh'
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


#==============================================================
# This script is supposed to be run on the login shell of triton with bash.
# This creates directories & input data files for each line
# specified in 'multiple_runs_data/multiple_runs_params.txt'.
#
# Parameters other than the ones specified in the file are taken from
# the default input data file.
#
# After creating the directories & files, this script submits
# an array of jobs using a temporary script created at the end
# of this file. It's not saved anywhere to avoid the risk of
# accidentally using it for running the program.
#
# The directories &c. could also be created by each of the jobs
# in the array, but an external script that does it is there for
# two reasons:
# 1. The binary needs to be built centrally in any case
# 2. 'jq' didn't seem to be available on the nodes
#
# (jq is used to modify the .json files per run)
#==============================================================


#==============================================================
# Variables
#==============================================================

PROJECT_ROOT_DIR="$WRKDIR/cuda_bubble"
MUL_RUN_DIR="$PROJECT_ROOT_DIR/multiple_runs_data"
PARAMS_FILE="$MUL_RUN_DIR/multiple_runs_params.txt"
BINARY="$PROJECT_ROOT_DIR/bin/cubble"
INPUT_DATA="input_data.json"
SAVE_DATA="save.json"
NUM_RUNS=-1


#==============================================================
# Create a directory for each run and modify the input_data.json
# with the arguments given in 'multiple_run_data/parameters.txt.
#
# Each line in the parameter file corresponds to one job.
# Each line should list all the non-default parameter names
# and values for one job separated with ' '.
#
# "PhiTarget 0.7 AvgRad 0.05 Kappa 0.01" would mean
# that one job uses those three specific values for those three
# parameters and all the rest are taken from the default input
# data file.
#==============================================================

cd $PROJECT_ROOT_DIR
make clean

while read LINE
do
  JQ_FILTER=''
  JQ_ARGS=''
  i=0
  
  for WORD in $LINE
  do    
    if [ $((i % 2)) == 0 ]
    then
      JQ_ARGS="$JQ_ARGS --argjson $WORD"
      JQ_FILTER="$JQ_FILTER .$WORD = \$$WORD | "
    else
      JQ_ARGS="$JQ_ARGS $WORD"
    fi
    
    i=$((i + 1))
  done
  
  # Remove the trailing " | "
  JQ_FILTER="${JQ_FILTER: : -3}"
  NUM_RUNS=$((NUM_RUNS + 1))

  mkdir -p $MUL_RUN_DIR/$NUM_RUNS/data
  jq $JQ_ARGS '"$JQ_FILTER"' $PROJECT_ROOT_DIR/data/$INPUT_DATA > $MUL_RUN_DIR/$NUM_RUNS/data/$INPUT_DATA

done < $PARAMS_FILE


#==============================================================
# Create temporary shell scripts for submitting the runs with sbatch.
#==============================================================

NEWLINE=$'\n'
BUILD_SCRIPT="#!/bin/sh"${NEWLINE}"module purge"${NEWLINE}"module load goolfc/triton-2017a"${NEWLINE}"cd $PROJECT_ROOT_DIR"${NEWLINE}"srun make final"${NEWLINE}
ARRAY_SCRIPT="#!/bin/sh"${NEWLINE}"module purge"${NEWLINE}"module load goolfc/triton-2017a"${NEWLINE}"cd $MUL_RUN_DIR/\$SLURM_ARRAY_TASK_ID"${NEWLINE}"cat data/$INPUT_DATA"${NEWLINE}
#"srun $BINARY $INPUT_DATA $SAVE_DATA"${NEWLINE}

#printf "$ARRAY_SCRIPT"

BUILD_JOB_ID=$(printf "$BUILD_SCRIPT" | sbatch --mem=1G --time=04:00:00 --gres=gpu:1 --constraint=pascal --mail-user=juhana.lankinen@aalto.fi --mail-type=ALL | tail -1 | awk -v N=4 '{print $N}')
printf "$ARRAY_SCRIPT" | sbatch --mem=1G --time=04:00:00 --gres=gpu:1 --constraint=pascal --mail-user=juhana.lankinen@aalto.fi --mail-type=ALL --dependency=aftercorr:"$BUILD_JOB_ID" --array=0-$NUM_RUNS
