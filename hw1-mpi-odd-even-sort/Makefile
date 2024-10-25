CC = mpicc
CXX = mpicxx
CXXFLAGS = -std=c++17 -O3 -lm -Wall -Wextra -march=native
CXXFLAGS += -DNO_FINALIZE
CFLAGS = -O3 -lm
TARGETS = hw1

ifeq ($(PROFILING), 1)
	CXXFLAGS += -DPROFILING
endif

ifeq ($(DEBUG), 1)
	CXXFLAGS += -DDEBUG
endif

ifeq ($(SANITIZE), 1)
	CXXFLAGS += -g
endif

ifeq ($(SANITIZE), 1)
	CXXFLAGS += -fsanitize=address -fsanitize=undefined
endif

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS)
