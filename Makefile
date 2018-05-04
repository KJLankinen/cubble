BIN_DIR = bin
SRC_PATH = src/
DATA_PATH = data/

PRG = Main.cpp
OBJS = Integrator.o

CC = g++
COMMON_FLAGS = -Wall -std=c++11 -DDATA_PATH="$(DATA_PATH)"
OPTIM_FLAGS = -O3
FLAGS = $(COMMON_FLAGS) $(OPTIM)

.PHONY: all
all: cubble

cubble: $(PRG) $(OBJS)
	$(CC) $(PRG) $(OBJS) $(FLAGS) -o $(BIN_DIR)/cubble

$(PRG):
	$(eval PRG = $(SRC_PATH)$(PRG))

$(OBJS): $(BIN_DIR)/%.o : $(SRC_PATH)%.cpp
	@mkdir $(BIN_DIR)
	$(CC) $(FLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -r $(BIN_DIR)
