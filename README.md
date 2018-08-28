# CUBBLE (CUDA bubble)

## To Whom It May Concern,

This is a CUDA accelerated version of and older code for simulating the coarsening and mechanics of bubbles. Everything has been rewritten from scratch, meaning the old version was only used as a model of the simulation.

In this readme, the program and its parts are introduced. The code doesn't contain a lot of comments, but the functions and variables should be (in most cases) well enough named that their purpose becomes clear.

## Contents of the repository
In addition to this readme, the repository contains the following items:
- **bin/**: This is where the binaries of the code are compiled into.
- **cuda_bubble.sh**: Script for slurm (triton).
- **data/**: This contains two .json files (input and output parameters). The data written by the program is also saved here.
- **include/**: All the extra dependencies are put here. E.g. the json parser and cub related files.
- **Makefile**: Used to build the program.
- **scripts/**: Contains some more or less useful scripts and code snipptes.
- **src/**: Contains all of the source code.

## Description of the source files and their purposes
- **Bubble.h**: A convenience wrapper around data & stream output operator. Contains nothing of real interest, but is useful for easy printing of data to file/cout.
- **Cell.h**: A convenience wrapper around some data. Should be removed & the relying code reworked.
- **CubbleApp.h/.cpp**: 'The application'. Separate from the entry point to the program, which is in Main. Basically initializes, runs and cleans up after the main simulation.
- **CudaConainer.h**: A wrapper around raw data. This was used extensively in an earlier version of the program. This should be removed and the relying code reworked accordingly.
- **DeviceMemoryHandler.h/.cpp**: This file controls the allocation and deallocation of device memory. This is potentially a very dangerous place to play around, since it exposes raw pointers to device memory. Keep your fingers away, unless you understand what it does and have a good idea of what you're doing. Basically if you feel comfortable with pointer arithmetic and C-level memory handling, feel free to change and add things.
- **Env.h**: This is basically a wrapper around the parameters of the program. All the input/output parameters from the .json files are handled by this. Also contains some program constants and other useful data, which should only be defined in one location. CubbleApp creates a shared pointer of this and it's passed around in constructor to anyone who needs to get/set parameters during the simulation. Whenever new input parameters are needed, add them to the input .json file _AND_ here.
- **Fileio.h**: Contains some file IO functions.
- **Macros.h**: Contains convenience macros. Preprocessor constants set in Makefile affect the definitions of the macros.
- **Main.cpp**: Contains the entry point of the program.
- **Simulator.h/.cu**: These files contain the simulation proper. All the CUDA code can be found from these files. This is where almost everything of interest happens, everything else mostly just supports these files.
- **Util.h**: Contains some general utility functions like assertations.
- **Vec.h**: An implementation of a 3D vector. Was used more extensively in an earlier version. Now this could be removed and the code that relies on this could be reworked.

## Building and running the program
**N.B.** The dimensionality of the simulation is controlled from the makefile.

The makefile contains multiple different commands for building the program:
```
make
make debug
make final
make clean
```

The default way is to just use 'make' in the folder that contains the makefile. This uses the current properties and builds a version which has some but not all debug capabilities turned on. For example assertations are defined in this version, which makes it easier to catch programming errors, but might make the program slightly less efficient.
The other two ways build different versions with either all or none of the debug options on. 'make clean' is used to delete the built binaries and temporary files created by emacs from source, script and data folders.

In addition to the options above, there are some extra parameters in the makefile which can be used to e.g. turn profiling on/off.

The program can be run by typing
```
make run
```
or by manually writing the path to the executable and the io files, e.g.
```
bin/cubble data.json save.json
```

## Notes and tips
The whole program was authored by one person, meaning it might not be the most easy or intuitive to use for someone else. If you as a user find something that was difficult for you to understand, but to which you were able to find an answer to, make note of it. If you have access to this repository, please add a small comment or a description of the problem and solution to it below. If you find that the implementation is stupid, feel free to make it less stupid.

As an example:
- When adding new parameters to the .json, always add an explanation of the parameter **BEFORE** the parameter itself: "EliteParamExpl" : "This parameter controls the eliteness of the program.", \n "EliteParam" : 1337.1337