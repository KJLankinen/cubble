BIN_DIR = bin
SRC_PATH = src/
DATA_PATH = data/

OBJ_NAMES := Bubble.o Integrator.o Cell.o
OBJS = $(foreach OBJ, $(OBJ_NAMES), $(BIN_DIR)/$(OBJ))
HEADERS := $(wildcard $(SRC_PATH)*.h)
EXEC = $(BIN_DIR)/cubble

CC := g++
COMMON_FLAGS := -Wall -std=c++14 -DDATA_PATH="$(DATA_PATH)"
OPTIM_FLAGS := -O3
FLAGS := $(COMMON_FLAGS) $(OPTIM)

.PHONY : all
all : $(EXEC)

$(EXEC) : $(SRC_PATH)Main.cpp $(OBJS) $(HEADERS)
	$(CC) $< $(OBJS) $(FLAGS) -o $@

$(BIN_DIR)/%.o : $(SRC_PATH)%.cpp
	@mkdir -p $(BIN_DIR)
	$(CC) $< $(FLAGS)-c -o $@

.PHONY : clean
clean :
	rm -fr $(BIN_DIR)
	rm -f $(SRC_PATH)*~
