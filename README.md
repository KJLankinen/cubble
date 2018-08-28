To Whom It May Concern,

This is a CUDA accelerated version of and older code for simulating the coarsening and mechanics of bubbles. Everything has been rewritten and the old code was only used as a model of the simulation.

In this readme, the program and its parts are introduced. The code doesn't contain a lot of comments, but the functions and variables should be (in most cases) well enough named that their purpose becomes clear.

Below, the source files are listed in alphabetical order and short description of each file and it's purpose is given.

Bubble.h
	A convenience wrapper around data & stream output operator. Contains nothing of real interest, but is useful for easy printing of data to file/cout.

Cell.h
	A convenience wrapper around some data. Should be removed & the relying code reworked.

CubbleApp.h/.cpp
	'The application'. Separate from the entry point to the program, which is in Main. Basically initializes, runs and cleans up after the main simulation.

CudaConainer.h
	A wrapper around raw data. This was used extensively in an earlier version of the program. This should be removed and the relying code reworked accordingly.

DeviceMemoryHandler.h/.cpp
	This file controls the allocation and deallocation of device memory. This is potentially a very dangerous place to play around, since it exposes raw pointers to device memory. Keep your fingers away, unless you understand what it does and have a good idea of what you're doing. Basically if you feel comfortable with pointer arithmetic and C-level memory handling, feel free to change and add things.

Env.h
	This is basically a wrapper around the parameters of the program. All the input/output parameters from the .json files are handled by this. Also contains some program constants and other useful data, which should only be defined in one location. CubbleApp creates a shared pointer of this and it's passed around in constructor to anyone who needs to get/set parameters during the simulation. Whenever new input parameters are needed, add them to the input .json file _AND_ here.

Fileio.h
	Contains some file IO functions.

Macros.h
	Contains convenience macros. Preprocessor constants set in Makefile affect the definitions of the macros.

Main.cpp
	Contains the entry point of the program.

Simulator.h/.cu
	These files contain the simulation proper. All the CUDA code can be found from these files. This is where almost everything of interest happens, everything else mostly just supports these files.

Util.h
	Contains some general utility functions like assertations.

Vec.h
	An implementation of a 3D vector. Was used more extensively in an earlier version. Now this could be removed and the code that relies on this could be reworked.