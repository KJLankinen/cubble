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

PARAMS_FILE="multiple_runs_data/params.txt"
NUM_RUNS=0

while read line
do
  echo "$line"
  NUM_RUNS += 1
done < $PARAMS_FILE

echo "sbatch scripts/array_run_dont_use_directly.sh --mem=1G --time=04:00:00 --gres=gpu:1 --constraint=pascal --array=0-$NUM_RUNS"
