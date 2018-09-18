#!/bin/bash

PARAM_FILE_PATH=$WRKDIR/cuda_bubble/multiple_runs_data

rm -f $PARAM_FILE_PATH/multiple_runs_params.txt

for i in {0..16}
do
printf "PhiTarget 0.$((900 - $((i * 25)))) RngSeed $RANDOM\n" >> $PARAM_FILE_PATH/multiple_runs_params.txt
done
