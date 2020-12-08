#!/bin/bash
module load OpenMPI/4.0.1-GCC-8.3.0-2.32
module load cuda/10.1.243

ulimit -c unlimited

/scratch/work/lankinj5/cubble/bin/optimized/cubble /scratch/work/lankinj5/cubble/input_parameters.json
