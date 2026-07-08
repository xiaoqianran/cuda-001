NVCC ?= nvcc
CUDA_PATH ?= /usr/local/cuda
all: fft_filter
fft_filter: src/fft_filter.cu
	$(NVCC) -arch=sm_75 -O3 -I$(CUDA_PATH)/include -L$(CUDA_PATH)/lib64 -lcufft -o $@ $<
clean:
	rm -f fft_filter
