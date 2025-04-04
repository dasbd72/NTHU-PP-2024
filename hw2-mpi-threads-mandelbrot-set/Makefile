CC = gcc
CXX = g++
LDLIBS = -lpng
CFLAGS = -lm -O3 -Wall -Wextra -march=native
CFLAGS += -DSIMD_ENABLED
CFLAGS += -DNO_FINALIZE
CFLAGS += -DPOOLING_ENABLED
hw2a: CFLAGS += -pthread -DMULTITHREADED=1
hw2b: CC = mpicc
hw2b: CXX = mpicxx
hw2b: CFLAGS += -fopenmp -DMULTITHREADED=2 -DMPI_ENABLED
TARGETS = hw2a hw2b

ifeq ($(VECTORIZE_VERBOSE), 1)
	CFLAGS += -fopt-info-vec-all
endif

ifeq ($(PROFILING), 1)
	CFLAGS += -DPROFILING
endif

ifeq ($(DEBUG), 1)
	CFLAGS += -DDEBUG
endif

ifeq ($(SANITIZE), 1)
	CFLAGS += -g
endif

ifeq ($(SANITIZE), 1)
	CFLAGS += -fsanitize=address
endif

CXXFLAGS = $(CFLAGS) -std=c++17

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o)
