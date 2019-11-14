# CUBBLE (CUDA bubble)

## To Whom It May Concern,

This is a CUDA accelerated version of and older code for simulating the coarsening and mechanics of bubbles. Everything has been rewritten from scratch, meaning the old version was only used as a model of the simulation.

## Contents of the repository
In addition to this readme, the repository contains the following items:
- **include/**: All the extra dependencies are put here. E.g. the json parser and cub related files.
- **makefile**: Used to build the program.
- **scripts/**: Contains scripts for running the program on triton and some scripts for plotting data.
- **src/**: Contains all of the source code.
- **default/**: Directory for the binaries of Make target 'default'
- **debug/**: Directory for the binaries of Make target 'debug'
- **final/**: Directory for the binaries of Make target 'final'

## Building and running the program
**N.B.** The dimensionality of the simulation is controlled from the makefile.

Each make target is built into a separate directory, final, default or debug. Each of these directories has its own makefile and all the targets are built/cleaned in the same way:
```
make
make clean
```

Final target is the one that should be run when doing simulations. It's the fastest and most optimized.

Default target is built with -O2 flag so it's quite fast, but some internal debug cababilities are still on and it's significantly slower than the final target. Mostly for testing some new cababilities.

Debug is built with -O0 and debug cababilities, only meant for debugging. Very slow.

In addition to the options above, there are some extra parameters in the makefile which can be used to e.g. turn profiling on/off.

The program can be run by typing
```
make run
```
or by manually writing the path to the executable and the io files, e.g.
```
final/bin/cubble input_parameters.json state.bin
```
The program runs until a certain amount of bubbles is left. After this, the program writes one final data file and returns.

**N.B.** The parameter that controls this amount of bubbles (called MinNumBubbles) should always be larger than the number of bubbles in one cell multiplied by 3^NumDim. In other words, if the number of bubbles in a cell is 32 and the dimensionality of the program is 2 (2D simulation), then the minimum number of bubbles should be larger than **32 * 3^2 = 32 * 3 * 3 = 288**. For 3D this would be 864. **300 and 900 are nice round numbers for MinNumBubbles**.

The reason for this is that the neighbor search is done in a manner that assumes at least 3 cells in each dimension. If there are less than 3 cells per dimension, some cells are searched through more than once, leading to bubbles having the same bubble as a neighbor multiple times. The implementation should and could be improved to circumvent this, but "in the mean time" just follow the above rule.

## Notes and tips
The whole program was authored by one person, meaning it might not be the most easy or intuitive to use for someone else. If you as a user find something that was difficult for you to understand, but to which you were able to find an answer to, make note of it. If you have access to this repository, please add a small comment or a description of the problem and solution to it below. If you find that the implementation is stupid, feel free to make it less stupid.

- When adding new parameters to the .json, always add an explanation of the parameter **BEFORE** the parameter itself: "ParamExpl" : "This parameter means this and that.", \n "Param" : 3.1415
