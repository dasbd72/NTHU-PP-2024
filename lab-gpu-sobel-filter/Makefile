NVCC = nvcc 
NVCCFLAGS = -O3 -std=c++11 -Xptxas=-v -arch=sm_61 --use_fast_math -DSOBEL_SMEM_ENABLED

ifeq ($(PROFILING), 1)
	NVCCFLAGS += -DPROFILING
endif

HIPCC = hipcc
HIPCCFLAGS  = -O3 -std=c++11 --offload-arch=gfx90a

LDFLAGS = -lpng -lz

TARGETS = sobel sobel-amd

sobel: sobel.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ $<

sobel-amd: sobel.hip
	$(HIPCC) $(HIPCCFLAGS) $(LDFLAGS) -o $@ $<

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS)
