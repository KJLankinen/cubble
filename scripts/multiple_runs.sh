#!/bin/bash

#SBATCH --mem=1G
#SBATCH --time=04:00:00						## wallclock time hh::mm::ss
#SBATCH --gres=gpu:1 --constraint='pascal'			## P100
#SBATCH --mail-user=juhana.lankinen@aalto.fi --mail-type=ALL

PROJECT_ROOT_DIR="$WRKDIR/cuda_bubble"
MUL_RUN_DIR="$PROJECT_ROOT_DIR/multiple_runs_data"
PARAMS_FILE="$MUL_RUN_DIR/params.txt"
JQ_ARG_PREFIX="--argjson"

module purge
module load goolfc/triton-2017a

cd $PROJECT_ROOT_DIR
srun make clean
srun make final

NUM_RUNS=-1

cd $MUL_RUN_DIR

while read LINE
do
  JQ_COMMANDS=''
  JQ_ARGS=''
  i=0
  for WORD in $LINE
  do
    if [ $((i % 2)) == 0 ]
    then
      JQ_ARGS="$JQ_ARGS $JQ_ARG_PREFIX $WORD"
      JQ_COMMANDS="$JQ_COMMANDS .$WORD = \$$WORD | "
    else
      JQ_ARGS="$JQ_ARGS $WORD"
    fi
    i=$((i + 1))
  done
  
  # Remove the trailing " | "
  JQ_COMMANDS="${JQ_COMMANDS: : -3}"
  NUM_RUNS=$((NUM_RUNS + 1))

  mkdir $NUM_RUNS $NUM_RUNS/data
  jq $JQ_ARGS '$JQ_COMMANDS' $WRKDIR/cuda_bubble/data/input_data.json > $WRKDIR/cuda_bubble/multiple_runs/$NUM_RUNS/data/input_data.json
done < $PARAMS_FILE

echo "sbatch scripts/array_run_dont_use_directly.sh --mem=1G --time=04:00:00 --gres=gpu:1 --constraint=pascal --array=0-$NUM_RUNS"
