NVCC = nvcc 
NVCXX = nvc++
NVCCFLAGS = -O3 -std=c++11 -Xptxas=-v -arch=sm_61
NVCXXFLAGS = -O3 -std=c++14  -Xptxas=-v -tp=cascadelake -acc -gpu=cc60
MNIST=-I/share/testcases/lab-mnist/weights/mnist/include

ifeq ($(PROFILING), 1)
	NVCCFLAGS += -DPROFILING
endif

HIPCC = hipcc
HIPCCFLAGS  = -O3 -std=c++11 --offload-arch=gfx90a

LDFLAGS = -lpng -lz

TARGETS = sobel sobel-amd mnist

sobel: sobel.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ $<

sobel-amd: sobel.hip
	$(HIPCC) $(HIPCCFLAGS) $(LDFLAGS) -o $@ $<

mnist: mnist.cpp
	$(NVCXX) $(NVCXXFLAGS) $(MNIST) -o $@ $<

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS)
