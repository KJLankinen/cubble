BIN_DIR = bin
SRC_PATH = src/
DATA_PATH = data/

PRG = $(SRC_PATH)Main.cpp
OBJ_NAMES := Integrator.o
OBJS := $(foreach OBJ, $(OBJ_NAMES), $(BIN_DIR)/$(OBJ))

HEADERS = $(wildcard $(SRC_PATH)*.h)

CC = g++
COMMON_FLAGS = -Wall -std=c++11 -DDATA_PATH="$(DATA_PATH)"
OPTIM_FLAGS = -O3
FLAGS = $(COMMON_FLAGS) $(OPTIM)

.PHONY: all
all: cubble

cubble: $(PRG) $(OBJS)
	$(CC) $(PRG) $(OBJS) $(FLAGS)-o $(BIN_DIR)/cubble

$(BIN_DIR)/%.o : $(SRC_PATH)%.cpp $(HEADERS)
	@mkdir -p $(BIN_DIR)
	$(CC) $< $(FLAGS)-c -o $@

.PHONY: clean
clean:
	rm -fr $(BIN_DIR)
