BIN_DIR = bin
SRC_PATH = src/
DATA_PATH = data/

OBJ_NAMES := Integrator.o
OBJS = $(foreach OBJ, $(OBJ_NAMES), $(BIN_DIR)/$(OBJ))
HEADERS := $(wildcard $(SRC_PATH)*.h)
EXEC = $(BIN_DIR)/cubble

NUM_DIM := 2

CC := g++
COMMON_FLAGS := -Wall -std=c++14 -DDATA_PATH="$(DATA_PATH)" -DNUM_DIM=$(NUM_DIM)
OPTIM_FLAGS := -O2
FLAGS := $(COMMON_FLAGS) $(OPTIM_FLAGS)

.PHONY : all
all : $(EXEC)

.PHONY : set_debug_flags
set_debug_flags :
	$(eval OPTIM_FLAGS = -O0 -g3 -p)
	$(eval FLAGS = $(COMMON_FLAGS) $(OPTIM_FLAGS))

.PHONY : set_final_flags
set_final_flags :
	$(eval OPTIM_FLAGS = -O3 -DNDEBUG)
	$(eval FLAGS = $(COMMON_FLAGS) $(OPTIM_FLAGS))

.PHONY : debug
debug : set_debug_flags $(EXEC)

.PHONY : final
final : set_final_flags $(EXEC)

$(EXEC) : $(SRC_PATH)Main.cpp $(OBJS) $(HEADERS)
	$(CC) $< $(OBJS) $(FLAGS) -o $@

$(BIN_DIR)/%.o : $(SRC_PATH)%.cpp
	@mkdir -p $(BIN_DIR)
	$(CC) $< $(FLAGS) -c -o $@

.PHONY : clean
clean :
	rm -fr $(BIN_DIR)
	rm -f $(SRC_PATH)*~
