BIN_PATH = bin/
SRC_PATH = src/
DATA_PATH = data/

PRG = $(SRC_PATH)Main.cpp

CC = g++
COMMON_FLAGS = -Wall -std=c++11
OPTIM_FLAGS = -O3
FLAGS = $(COMMON_FLAGS) $(OPTIM)

.PHONY: all
all: prog

prog: $(PRG)
	$(CC) $(PRG) $(FLAGS) -o $(BIN_PATH)cubble

.PHONY: clean
clean:
	rm -f $(BIN_PATH)*
