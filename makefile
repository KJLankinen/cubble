# Directories
# -----------------------------------------------------
# bin will be created when building the program.
# Executable and the intermediate object files will be placed there.
# 'make clean' will completely remove bin and its contents.
BIN_PREFIX := bin

# All the source files recide here.
SRC_PATH := src

# Included (external) headers
INCL = -Iincl/

# Object files, headers and the main executable
# -----------------------------------------------------
# List all objects that contain CPU code.
OBJ_NAMES := Main.o Simulator.o Kernels.o
OBJS = $(addprefix $(BIN_PATH)/, $(OBJ_NAMES))

# List all the objects that contain GPU code.
# Overlap with the objects above is totally fine.
# These are only related to linking, compiling is done automatically
# based on the file extension (.cpp vs. .cu)
GPU_OBJ_NAMES := Simulator.o Kernels.o
GPU_OBJS = $(addprefix $(BIN_PATH)/, $(GPU_OBJ_NAMES))

# Find all headers in source dir.
HEADERS := $(wildcard $(SRC_PATH)/*.h)
HEADERS += $(wildcard $(SRC_PATH)/*.cuh)

# Name of the linked GPU code.
GPU_CODE = $(BIN_PATH)/GPUCode.o

# Name of the final executable.
EXEC = $(BIN_PATH)/cubble

# Compilers to use
# -----------------------------------------------------
CPU_COMPILER := g++
GPU_COMPILER := nvcc

# External libraries to link
# -----------------------------------------------------
LIBS := -lcudart -lcurand -lnvToolsExt -lpthread

# Flags
# -----------------------------------------------------
#  PROFILE=-DPROFILE can be passed to make from cmd, when profiling
PROFILE ?=
LINK_FLAGS ?=
NVCC_DEFINES := -D_FORCE_INLINES -D_MWAITXINTRIN_H_INCLUDED -D__STRICT_ANSI__
override CPU_FLAGS += -Wall -std=c++14 -m64
GPU_ARCH := -gencode arch=compute_70,code=compute_70	\
	    -gencode arch=compute_70,code=sm_70
override GPU_FLAGS += $(GPU_ARCH) $(NVCC_DEFINES) \
    -std=c++14 --expt-relaxed-constexpr
COMMON_FLAGS = $(INCL) $(OPTIM_FLAGS) $(DEFINES)

# Default first rule
# -----------------------------------------------------
.PHONY : all
all :
	$(eval BIN_PATH = $(BIN_PREFIX)/optimized)
	mkdir -p $(BIN_PATH)
	$(MAKE) -j8 BIN_PATH=$(BIN_PATH) GPU_FLAGS=-lineinfo OPTIM_FLAGS=-O3 \
	    DEFINES='-DNDEBUG $(PROFILE)' $(EXEC)

# Debug rule
# -----------------------------------------------------
.PHONY : debug
debug :
	$(eval BIN_PATH = $(BIN_PREFIX)/debug)
	mkdir -p $(BIN_PATH)
	$(MAKE) -j8 BIN_PATH=$(BIN_PATH) GPU_FLAGS='-g -G' \
	    OPTIM_FLAGS=-O0 LINK_FLAGS=-g3 $(EXEC)

# Rule for main executable.
# -----------------------------------------------------
$(EXEC) : $(HEADERS) $(OBJS) $(GPU_CODE)
	$(CPU_COMPILER) $(LINK_FLAGS) $(OBJS) $(GPU_CODE) $(LIBS) -o $@

# Rule for linking the GPU code to a single object file
# -----------------------------------------------------
$(GPU_CODE) : $(GPU_OBJS)
	$(GPU_COMPILER) $(GPU_ARCH) -dlink $^ -o $@


# Rule for the intermediate objects
# -----------------------------------------------------
# CPU code
$(BIN_PATH)/%.o : $(SRC_PATH)/%.cpp
	$(CPU_COMPILER) $< $(CPU_FLAGS) $(COMMON_FLAGS) -c -o $@

# GPU code
$(BIN_PATH)/%.o : $(SRC_PATH)/%.cu
	$(GPU_COMPILER) $< $(GPU_FLAGS) $(COMMON_FLAGS) -dc -o $@

# Clean up
# -----------------------------------------------------
.PHONY : clean
clean :
	rm -rf $(BIN_PREFIX)
