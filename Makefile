# -----------------------------------------------------
# Directories
# -----------------------------------------------------
BIN_DIR = bin
SRC_PATH = src/
DATA_PATH = data/

# -----------------------------------------------------
# Object files, headers and the main executable
# -----------------------------------------------------
OBJ_NAMES := Simulator.o BubbleManager.o
OBJS = $(foreach OBJ, $(OBJ_NAMES), $(BIN_DIR)/$(OBJ))
HEADERS := $(wildcard $(SRC_PATH)*.h)
EXEC = $(BIN_DIR)/cubble

# -----------------------------------------------------
# How many dimensions to simulate.
# The compiled code is different for different dimensions,
# so this can't be given as a normal program parameter.
# -----------------------------------------------------
NUM_DIM := 2

# -----------------------------------------------------
# Compiler to use
# -----------------------------------------------------
CCPU := g++
CGPU := nvcc

# -----------------------------------------------------
# External libraries to link to
# -----------------------------------------------------
LIB := #-lcudart -lcurand

# -----------------------------------------------------
# Preprocessor defines
# -----------------------------------------------------
DEFINES := -DDATA_PATH="$(DATA_PATH)" -DNUM_DIM=$(NUM_DIM)

# -----------------------------------------------------
# Flags
# -----------------------------------------------------
CPU_FLAGS := -Wall
GPU_FLAGS := -arch=sm_20 -dlink
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
$(EXEC) : $(SRC_PATH)Main.cpp $(OBJS) $(HEADERS)
	$(CCPU) $< $(OBJS) $(CPU_FLAGS) $(COMMON_FLAGS) $(OPTIM_FLAGS) $(DEFINES) $(LIB) -o $@

# -----------------------------------------------------
# Rule for the intermediate objects
# -----------------------------------------------------
$(BIN_DIR)/%.o : $(SRC_PATH)%.cpp
	@mkdir -p $(BIN_DIR)
	$(CCPU) $< $(CPU_FLAGS) $(COMMON_FLAGS) $(OPTIM_FLAGS) $(DEFINES) $(LIB) -c -o $@

# -----------------------------------------------------
# Clean up
# -----------------------------------------------------
.PHONY : clean
clean :
	rm -fr $(BIN_DIR)
	rm -f $(SRC_PATH)*~
