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
# Profile or not?
# -----------------------------------------------------

USE_PROFILING := 0


# -----------------------------------------------------
# Directories
# -----------------------------------------------------

# bin will be created when building the program.
# Executable and the intermediate object files will be placed there.
# 'make clean' will completely remove bin and its contents.
BIN_PATH := bin/

# All the source files recide here.
SRC_PATH := src/

# Included (external) headers
INCL = -Iinclude/ 


# -----------------------------------------------------
# Object files, headers and the main executable
# -----------------------------------------------------

# List all objects that contain CPU code.
OBJ_NAMES := Main.o CubbleApp.o Simulator.o UtilityKernels.o BubbleKernels.o IntegrationKernels.o
OBJS = $(foreach OBJ, $(OBJ_NAMES), $(BIN_PATH)$(OBJ))

# List all the objects that contain GPU code.
# Overlap with the objects above is totally fine.
# These are only related to linking, compiling is done automatically
# based on the file extension (.cpp vs. .cu)
GPU_OBJ_NAMES := BubbleKernels.o Simulator.o UtilityKernels.o IntegrationKernels.o
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

LIBS := -lcudart -lcurand -lnvToolsExt


# -----------------------------------------------------
# Preprocessor defines
# -----------------------------------------------------

DEFINES := -DNUM_DIM=$(NUM_DIM) -DUSE_PROFILING=$(USE_PROFILING) -DPBC_X=$(PBC_X) -DPBC_Y=$(PBC_Y) -DPBC_Z=$(PBC_Z)
NVCC_DEFINES := -D_FORCE_INLINES -D_MWAITXINTRIN_H_INCLUDED -D__STRICT_ANSI__


# -----------------------------------------------------
# Flags
# -----------------------------------------------------

CUDA_ARCH := sm_60
CPU_FLAGS := -Wall -std=c++14 -m64
GPU_FLAGS := -arch=$(CUDA_ARCH) -std=c++11 -lineinfo --expt-relaxed-constexpr
COMMON_FLAGS :=  $(INCL)
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
	$(EXEC) input_parameters.json output_parameters.json

# -----------------------------------------------------
# Debug build, slow with all safety nets at place.
# -----------------------------------------------------

.PHONY : debug
debug : set_debug_flags $(EXEC)

.PHONY : set_debug_flags
set_debug_flags :
	$(eval OPTIM_FLAGS = -O0)
	$(eval CPU_FLAGS += -g3 -p)
	$(eval GPU_FLAGS += -g -G)


# -----------------------------------------------------
# Optimized build with no safety nets.
# -----------------------------------------------------

.PHONY : final
final : set_final_flags $(EXEC)

.PHONY : set_final_flags
set_final_flags :
	$(eval OPTIM_FLAGS = -O3)
	$(eval DEFINES += -DNDEBUG)


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
	@mkdir -p $(BIN_PATH)
	$(C_GPU) -arch=$(CUDA_ARCH) -dlink $^ -o $@


# -----------------------------------------------------
# Rule for the intermediate objects
# -----------------------------------------------------

# CPU code
$(BIN_PATH)%.o : $(SRC_PATH)%.cpp
	@mkdir -p $(BIN_PATH)
	$(eval OPTIONS = $(CPU_FLAGS) $(COMMON_FLAGS) $(OPTIM_FLAGS) $(DEFINES))
	$(C_CPU) $< $(OPTIONS) -c -o $@

# GPU code
$(BIN_PATH)%.o : $(SRC_PATH)%.cu
	@mkdir -p $(BIN_PATH)
	$(eval OPTIONS = $(GPU_FLAGS) $(COMMON_FLAGS) $(OPTIM_FLAGS) $(DEFINES) $(NVCC_DEFINES))
	$(C_GPU) $< $(OPTIONS) -dc -o $@


# -----------------------------------------------------
# Clean up
# -----------------------------------------------------

.PHONY : clean
clean :
	rm -f $(BIN_PATH)*
