NVCC ?= nvcc
CUDA_ARCH ?= -arch=sm_75
all: bilateral
bilateral: src/bilateral.cu
	$(NVCC) $(CUDA_ARCH) -O3 -o $@ $<
clean:
	rm -f bilateral
.PHONY: all clean
