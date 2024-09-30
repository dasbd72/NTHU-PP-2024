CC = gcc
CXX = g++
CFLAGS = -lm -O3 -Wall -Wextra -march=native -DSIMD_ENABLED
lab2_pthread: CFLAGS += -pthread
lab2_pthread: CFLAGS += -DMULTITHREADED=1
lab2_omp: CFLAGS += -fopenmp
lab2_omp: CFLAGS += -DMULTITHREADED=2
lab2_hybrid: CC = mpicc
lab2_hybrid: CXX = mpicxx
lab2_hybrid: CFLAGS += -fopenmp
lab2_hybrid: CFLAGS += -DMULTITHREADED=1 -DMPI_ENABLED
TARGETS = lab2_pthread lab2_omp lab2_hybrid

ifeq ($(VECTORIZE_VERBOSE), 1)
	CFLAGS += -fopt-info-vec-all
endif

ifeq ($(TIMING), 1)
	CFLAGS += -DTIMING
endif

ifeq ($(DEBUG), 1)
	CFLAGS += -DDEBUG
endif

ifeq ($(SANITIZE), 1)
	CFLAGS += -g
endif

ifeq ($(SANITIZE), 1)
	CFLAGS += -fsanitize=address -fsanitize=undefined
endif

CXXFLAGS = $(CFLAGS) -std=c++17

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o)
