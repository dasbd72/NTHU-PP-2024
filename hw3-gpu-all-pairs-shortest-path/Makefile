CFLAGS = -O3 -fopenmp
CXXFLAGS = -std=c++11 -O3 -fopenmp -march=native -mtune=native -Wall -Wextra
NVFLAGS = -std=c++11 -O3 -Xptxas="-v" -arch=sm_61
HIPCCFLAGS = -std=c++11 -O3 --offload-arch=gfx90a
LDFLAGS = -lm

hw3-1: CXXFLAGS += -pthread -DMANUAL_SIMD -DNO_FINALIZE
hw3-2: NVFLAGS += -Xcompiler -fopenmp
hw3-3: NVFLAGS += -Xcompiler -fopenmp
hw3-2-amd: HIPCCFLAGS += -fopenmp
hw3-3-amd: HIPCCFLAGS += -fopenmp

EXES = hw3-1 hw3-2 hw3-3

.PHONY: all clean

all: $(EXES)

clean:
	rm -f $(EXES)

seq: seq.cc
	g++ $(CXXFLAGS) -o $@ $?

hw3-1: hw3-1.cc
	g++ $(CXXFLAGS) $(LDFLAGS) -o $@ $?

hw3-2: hw3-2.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?

hw3-2-amd: hw3-2.hip
	hipcc $(HIPCCFLAGS) $(LDFLAGS) -o $@ $<

hw3-3: hw3-3.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?

hw3-3-amd: hw3-3.hip
	hipcc $(HIPCCFLAGS) $(LDFLAGS) -o $@ $<
