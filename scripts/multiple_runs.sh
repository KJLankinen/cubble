#!/bin/bash

#==============================================================
# Specify the arguments needed to run this particular script
#==============================================================

#SBATCH --mem=1G
#SBATCH --time=04:00:00						## wallclock time hh::mm::ss
#SBATCH --gres=gpu:1 --constraint='pascal'			## P100
#SBATCH --mail-user=juhana.lankinen@aalto.fi --mail-type=ALL


#==============================================================
# Variables
#==============================================================

PROJECT_ROOT_DIR="$WRKDIR/cuda_bubble"
MUL_RUN_DIR="$PROJECT_ROOT_DIR/multiple_runs_data"
PARAMS_FILE="$MUL_RUN_DIR/multiple_runs_params.txt"
NUM_RUNS=-1


#==============================================================
# Load needed modules
#==============================================================

#module purge
#module load goolfc/triton-2017a


#==============================================================
# Build the program
#==============================================================

cd $PROJECT_ROOT_DIR
#srun make clean
#srun make final


#==============================================================
# Create a directory for each run and modify the input_data.json
# with the arguments given in 'multiple_run_data/parameters.txt.
# Each line corresponds to one run.
#==============================================================

while read LINE
do
  JQ_FILTER=''
  JQ_ARGS=''
  i=0
  
  for WORD in $LINE
  do    
    # Even words are names of variables, odd words are the values
    # i.e. if the line is e.g. "PhiTarget 0.7 AvgRad 0.05 Kappa 0.01"
    # 'PhiTarget', 'AvgRad' and 'Kappa' are the names and
    # '0.7', '0.05' and '0.01' are the values.

    if [ $((i % 2)) == 0 ]
    then
      JQ_ARGS="$JQ_ARGS --argjson $WORD"
      JQ_FILTER="$JQ_FILTER.$WORD = \$$WORD | "
    else
      JQ_ARGS="$JQ_ARGS $WORD"
    fi
    
    i=$((i + 1))
  done
  
  # Remove the trailing " | "
  JQ_FILTER="${JQ_FILTER: : -3}"
  NUM_RUNS=$((NUM_RUNS + 1))

  # Easier to notice separate 'mkdir' commands when on separate lines
  echo "mkdir $MUL_RUN_DIR/$NUM_RUNS"
  echo "mkdir $MUL_RUN_DIR/$NUM_RUNS/data"
  echo "jq $JQ_ARGS '$JQ_FILTER' $PROJECT_ROOT_DIR/data/input_data.json > $MUL_RUN_DIR/$NUM_RUNS/data/input_data.json"

done < $PARAMS_FILE


#==============================================================
# Put an array of jobs to the queue
#==============================================================

echo "sbatch $PROJECT_ROOT_DIR/scripts/array_job_dont_use_this_directly.sh --mem=1G --time=04:00:00 --gres=gpu:1 --constraint=pascal --array=0-$NUM_RUNS"
