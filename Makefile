# -----------------------------------------------------
# Directories
# -----------------------------------------------------
BIN_PATH = bin/
SRC_PATH = src/
DATA_PATH = data/

# -----------------------------------------------------
# Object files, headers and the main executable
# -----------------------------------------------------
# List all objects that contain CPU code.
OBJ_NAMES := Simulator.o BubbleManager.o Test.o
OBJS = $(foreach OBJ, $(OBJ_NAMES), $(BIN_PATH)$(OBJ))

# List all the objects that contain GPU code.
# Overlap with the objects above is totally fine.
GPU_OBJ_NAMES := Test.o
GPU_OBJS = $(foreach OBJ, $(GPU_OBJ_NAMES), $(BIN_PATH)$(OBJ))

# Find all headers in source dir
HEADERS := $(wildcard $(SRC_PATH)*.h)

# Names of the linked GPU code and the final executable
GPU_CODE = $(BIN_PATH)gpuCode.o
EXEC = $(BIN_PATH)cubble

# -----------------------------------------------------
# How many dimensions to simulate.
# The compiled code is different for different dimensions,
# so this can't be given as a normal program parameter.
# -----------------------------------------------------
NUM_DIM := 2

# -----------------------------------------------------
# Compiler to use
# -----------------------------------------------------
C_CPU := g++
C_GPU := nvcc

# -----------------------------------------------------
# External libraries to link to
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
# First rule building the project with default settings
# -----------------------------------------------------
.PHONY : all
all : $(EXEC)

# -----------------------------------------------------
# Debug build
# -----------------------------------------------------
.PHONY : debug
debug : set_debug_flags $(EXEC)

.PHONY : set_debug_flags
set_debug_flags :
	$(eval OPTIM_FLAGS = -O0 -g3 -p)

# -----------------------------------------------------
# Optimized and 'reckless' build
# -----------------------------------------------------
.PHONY : final
final : set_final_flags $(EXEC)

.PHONY : set_final_flags
set_final_flags :
	$(eval OPTIM_FLAGS = -O3)
	$(eval DEFINES += -DNDEBUG)

# -----------------------------------------------------
# Rule for main executable
# -----------------------------------------------------
$(EXEC) : $(SRC_PATH)Main.cpp $(OBJS) $(GPU_CODE) $(HEADERS)
	$(C_CPU) $< $(OBJS) $(GPU_CODE) $(CPU_FLAGS) $(COMMON_FLAGS) $(OPTIM_FLAGS) $(DEFINES) $(LIBS) -o $@

# -----------------------------------------------------
# Rule for the gpu code
# -----------------------------------------------------
$(GPU_CODE) : $(GPU_OBJS)
	@mkdir -p $(BIN_PATH)
	$(C_GPU) -arch=sm_20 -dlink $^ -o $@

# -----------------------------------------------------
# Rule for the intermediate objects
# -----------------------------------------------------
$(BIN_PATH)%.o : $(SRC_PATH)%.cpp
	@mkdir -p $(BIN_PATH)
	$(C_CPU) $< $(CPU_FLAGS) $(COMMON_FLAGS) $(OPTIM_FLAGS) $(DEFINES) -c -o $@

$(BIN_PATH)%.o : $(SRC_PATH)%.cu
	@mkdir -p $(BIN_PATH)
	$(C_GPU) $< $(GPU_FLAGS) $(COMMON_FLAGS) $(OPTIM_FLAGS) $(DEFINES) -D_FORCE_INLINES -dc -o $@

# -----------------------------------------------------
# Clean up
# -----------------------------------------------------
.PHONY : clean
clean :
	rm -fr $(BIN_PATH)
	rm -f $(SRC_PATH)*~
