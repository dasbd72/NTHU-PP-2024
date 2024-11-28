NVCC = nvcc
NVFLAGS = -O3 -std=c++17 -Xptxas=-v -arch=sm_61 --use_fast_math

LDFLAGS = -lm

TARGETS = hw4

ifeq ($(PROFILING), 1)
	NVFLAGS += -DPROFILING
endif

all: $(TARGETS)

clean:
	rm -f $(TARGETS)

.PHONY: all clean

hw4: hw4.cu
	$(NVCC) $(NVFLAGS) -o $@ $? $(LDFLAGS)
