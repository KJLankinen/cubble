# -----------------------------------------------------
# Directories
# -----------------------------------------------------

# bin will be created when building the program.
# Executable and the intermediate object files will be placed there.
# 'make clean' will completely remove bin and its contents.
BIN_PATH = bin/

# All the source files recide here.
SRC_PATH = src/

# All the data files recide here.
DATA_PATH = data/


# -----------------------------------------------------
# Object files, headers and the main executable
# -----------------------------------------------------

# List all objects that contain CPU code.
OBJ_NAMES := Simulator.o BubbleManager.o CudaKernelsWrapper.o
OBJS = $(foreach OBJ, $(OBJ_NAMES), $(BIN_PATH)$(OBJ))

# List all the objects that contain GPU code.
# Overlap with the objects above is totally fine.
GPU_OBJ_NAMES := CudaKernelsWrapper.o
GPU_OBJS = $(foreach OBJ, $(GPU_OBJ_NAMES), $(BIN_PATH)$(OBJ))

# Find all headers in source dir.
HEADERS := $(wildcard $(SRC_PATH)*.h)

# Name of the linked GPU code.
GPU_CODE = $(BIN_PATH)gpuCode.o

# Name of the final executable.
EXEC = $(BIN_PATH)cubble


# -----------------------------------------------------
# How many dimensions to simulate.
# -----------------------------------------------------

# The compiled code is different for different dimensions,
# so this can't be given as a normal program parameter.
NUM_DIM := 2


# -----------------------------------------------------
# Compilers to use
# -----------------------------------------------------

C_CPU := g++
C_GPU := nvcc


# -----------------------------------------------------
# External libraries to link
# -----------------------------------------------------

LIBS := -lcudart -lcurand


# -----------------------------------------------------
# Preprocessor defines
# -----------------------------------------------------

DEFINES := -DDATA_PATH="$(DATA_PATH)" -DNUM_DIM=$(NUM_DIM)


# -----------------------------------------------------
# Flags
# -----------------------------------------------------

CPU_FLAGS := -Wall
GPU_FLAGS := -arch=sm_20
COMMON_FLAGS := -std=c++11
OPTIM_FLAGS := -O2


# -----------------------------------------------------
# First rule: Builds the project with default settings
# -----------------------------------------------------

.PHONY : all
all : $(EXEC)


# -----------------------------------------------------
# Debug build, slow with all safety nets at place.
# -----------------------------------------------------

.PHONY : debug
debug : set_debug_flags $(EXEC)

.PHONY : set_debug_flags
set_debug_flags :
	$(eval OPTIM_FLAGS = -O0 -g3 -p)


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
$(EXEC) : $(SRC_PATH)Main.cpp $(OBJS) $(GPU_CODE) $(HEADERS)
	$(eval OPTIONS = $(CPU_FLAGS) $(COMMON_FLAGS) $(OPTIM_FLAGS) $(DEFINES) $(LIBS))
	$(C_CPU) $< $(OBJS) $(GPU_CODE) $(OPTIONS) -o $@


# -----------------------------------------------------
# Rule for linking the GPU code to a single object file
# -----------------------------------------------------

$(GPU_CODE) : $(GPU_OBJS)
	@mkdir -p $(BIN_PATH)
	$(C_GPU) -arch=sm_20 -dlink $^ -o $@


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
	$(eval OPTIONS = $(GPU_FLAGS) $(COMMON_FLAGS) $(OPTIM_FLAGS) $(DEFINES))
	$(C_GPU) $< $(OPTIONS) -D_FORCE_INLINES -dc -o $@


# -----------------------------------------------------
# Clean up
# -----------------------------------------------------

.PHONY : clean
clean :
	rm -fr $(BIN_PATH)
	rm -f $(SRC_PATH)*~
