# CUBBLE
This is a C++/Cuda implementation of a wet foam model based on the work by
Durian. The program can be used to simulate two and three dimensional foams
in different conditions. More information of the physics of the simulation
can be found in [this thesis](http://urn.fi/URN:NBN:fi:hulib-202003301668)
writted about it.

## Contents of the repository
- **array.params**:
    File for controlling the inputs of an array job on the cluster
- **incl/**:
    External dependencies of the program
- **input_parameters.json**:
    File for controlling the inputs of the simulation
- **LICENSE**:
    The license of the program
- **makefile**:
    Makefile for building the program
- **README**:
    This file
- **scripts/**:
    Scripts for running the program on triton cluster and various scripts
    for plotting data
- **src/**:
    Source files

## Dependencies
- Linux environment
- GPU with compute capability 7.x or higher (e.g. Volta V100)
- Cuda 10.1.243 or higher
- gcc 6.5.0 or higher
- The contents of the incl directory

## Building the program
The program can be built with the accompanying makefile.
It contains three rules to choose from:
```
make
make debug
make clean
```
The first rule should normally be used to build the simulation. It builds
an optimized executable to `bin/optimized/cubble`. The second rule can be used
to build a very slow version of the code for debugging. Note that this version
is extremely slow compared to the optimized version, since the GPU and CPU are
synchronized between every GPU call and additionally all the compiler
optimizations are turned off. The debug executable can be found in
`bin/debug/cubble`. The third rule removes the contents of the bin directory.

## Running the program
Once the binary has been compiled, it can be found in `bin/xxx/cubble`, where
`xxx` depends on the build target. The program takes the path to the input file
as an argument. By default it's called `input_parameters.json`. Thus, running
the program is as simple as typing
```
bin/optimized/cubble input_parameters.json
```
assuming you've built the optimized version and are running the program from
the base directory.

## Inputs of the program
Read the comments inside the input file for more information on what each
input variable does. Adding new inputs is easy: just add them to the input
json and then access them from the source code. The source code does not
care about any extra variables in the input file, so new variables can be added
freely and they do nothing, if they're not accessed from the source code.

One should not change the names of the variables in the input file, unless
the names are changed in the source as well. In other words, the names of the
input variables are hard coded in the source code.

The best way to learn and understand what each variable does is to see where
it is used in the program, change it slightly and see how it affects the
simulation. Some care should be taken however, as some values are essentially
meaningless, like negative values for the physical parameters of the foam
(e.g. kappa, K, phi, etc.). Some values are also coupled with your running
environment, i.e. you should not save snapshots frequently if you have low disk
space or use very large simulation sizes (`bubbles.numStart` >= 10^7) if your
GPU doesn't have a lot of memory.

## Profiling the program
Profiling the program is an involved process and you should be sure to know
what you're doing before starting it. Make sure to read the available
documentation on Nsight-systems and Nsight-compute thoroughly before starting.
If `cudaProfilerApi` is to be used with profiling, one can pass the command line
argument `PROFILE=-DPROFILE` to `make` when compiling the optimized version of
the code. There are points in the source code between which profiling data is
collected, if the flag has been passed to the code. These can be moved around
at will, given that you know what you're doing.

## Debugging the program
You can use `cuda-gdg` to debug the program. For this you need to build the
program with the debug rule, i.e. `make debug`. Since the debug version of the
program is painfully slow, it can be time consuming to debug the simulation
if the input size is large. I recommend always checking if the problem or bug
manifests itself with a smaller problem size, e.g. 10^4 bubbles.
