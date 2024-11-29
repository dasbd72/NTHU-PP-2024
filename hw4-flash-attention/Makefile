NVCC = nvcc
NVFLAGS = -O3 -std=c++17 -Xptxas=-v -arch=sm_61 --use_fast_math
NVFLAGS += -DFLASH_ATTN=2

LDFLAGS = -lm

TARGETS = hw4

ifeq ($(PROFILING), 1)
	NVFLAGS += -DPROFILING
endif

ifeq ($(SANITIZE), 1)
	NVFLAGS += -Xcompiler -fsanitize=address -g -G
endif

all: $(TARGETS)

clean:
	rm -f $(TARGETS)

.PHONY: all clean

hw4: hw4.cu
	$(NVCC) $(NVFLAGS) -o $@ $? $(LDFLAGS)
