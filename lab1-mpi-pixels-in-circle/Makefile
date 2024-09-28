CC = mpicc
CXX = mpicxx
CFLAGS = -O3 -Wall -Wextra -march=native -DMPI_ENABLED
TARGETS = lab1

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
	rm -f $(TARGETS)
