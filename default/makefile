# -----------------------------------------------------
# How many dimensions to simulate.
# -----------------------------------------------------

NUM_DIM := 2


# -----------------------------------------------------
# Which dimensions use periodic boundary conditions?
# -----------------------------------------------------

PBC_X := 1
PBC_Y := 1
PBC_Z := 1

# -----------------------------------------------------
# Use flow or not?
# -----------------------------------------------------

USE_FLOW := 1

# -----------------------------------------------------
# Directories
# -----------------------------------------------------

# bin will be created when building the program.
# Executable and the intermediate object files will be placed there.
# 'make clean' will completely remove bin and its contents.
BIN_PATH := bin/

# All the source files recide here.
SRC_PATH := ../src/

# Included (external) headers
INCL = -I../incl/


# -----------------------------------------------------
# Object files, headers and the main executable
# -----------------------------------------------------

# List all objects that contain CPU code.
OBJ_NAMES := Main.o Simulator.o Kernels.o
OBJS = $(foreach OBJ, $(OBJ_NAMES), $(BIN_PATH)$(OBJ))

# List all the objects that contain GPU code.
# Overlap with the objects above is totally fine.
# These are only related to linking, compiling is done automatically
# based on the file extension (.cpp vs. .cu)
GPU_OBJ_NAMES := Simulator.o Kernels.o
GPU_OBJS = $(foreach OBJ, $(GPU_OBJ_NAMES), $(BIN_PATH)$(OBJ))

# Find all headers in source dir.
HEADERS := $(wildcard $(SRC_PATH)*.h)
HEADERS += $(wildcard $(SRC_PATH)*.cuh)

# Name of the linked GPU code.
GPU_CODE = $(BIN_PATH)GPUCode.o

# Name of the final executable.
EXEC = $(BIN_PATH)cubble

# -----------------------------------------------------
# Compilers to use
# -----------------------------------------------------

C_CPU := g++
C_GPU := nvcc


# -----------------------------------------------------
# External libraries to link
# -----------------------------------------------------

LIBS := -lcudart -lcurand -lnvToolsExt -lvtkCommonCore-8.0 -lvtkIOXML-8.0 -lvtkCommonDataModel-8.0


# -----------------------------------------------------
# Preprocessor defines
# -----------------------------------------------------

DEFINES := -DNUM_DIM=$(NUM_DIM) -DPBC_X=$(PBC_X) -DPBC_Y=$(PBC_Y) -DPBC_Z=$(PBC_Z) -DUSE_FLOW=$(USE_FLOW)
NVCC_DEFINES := -D_FORCE_INLINES -D_MWAITXINTRIN_H_INCLUDED -D__STRICT_ANSI__


# -----------------------------------------------------
# Flags
# -----------------------------------------------------

CUDA_ARCH := sm_60
CPU_FLAGS := -Wall -std=c++14 -m64
GPU_FLAGS := -arch=$(CUDA_ARCH) -std=c++11 --expt-relaxed-constexpr
COMMON_FLAGS := $(INCL)
OPTIM_FLAGS := -O2


# -----------------------------------------------------
# First rule: Builds the project with default settings
# -----------------------------------------------------

.PHONY : all
all : $(EXEC)

#-----------------------------------------------------
# Execute the program with default input files
#-----------------------------------------------------
.PHONY : run
run : $(EXEC)
	$(EXEC) ../input_parameters.json ../output_parameters.json

# -----------------------------------------------------
# Rule for main executable.
# -----------------------------------------------------

# By default has some safety nets at place but also uses some optimizations.
$(EXEC) : $(HEADERS) $(OBJS) $(GPU_CODE)
	$(C_CPU) $(OBJS) $(GPU_CODE) $(LIBS) -o $@


# -----------------------------------------------------
# Rule for linking the GPU code to a single object file
# -----------------------------------------------------

$(GPU_CODE) : $(GPU_OBJS)
	$(C_GPU) -arch=$(CUDA_ARCH) -dlink $^ -o $@


# -----------------------------------------------------
# Rule for the intermediate objects
# -----------------------------------------------------

# CPU code
$(BIN_PATH)%.o : $(SRC_PATH)%.cpp
	$(eval OPTIONS = $(CPU_FLAGS) $(COMMON_FLAGS) $(OPTIM_FLAGS) $(DEFINES))
	$(C_CPU) $< $(OPTIONS) -c -o $@

# GPU code
$(BIN_PATH)%.o : $(SRC_PATH)%.cu
	$(eval OPTIONS = $(GPU_FLAGS) $(COMMON_FLAGS) $(OPTIM_FLAGS) $(DEFINES) $(NVCC_DEFINES))
	$(C_GPU) $< $(OPTIONS) -dc -o $@


# -----------------------------------------------------
# Clean up
# -----------------------------------------------------

.PHONY : clean
clean :
	rm -f $(BIN_PATH)*
