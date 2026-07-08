NVCC?=nvcc
all: mini_pbrt
mini_pbrt: src/mini_pbrt.cu
	$(NVCC) -arch=sm_75 -O3 -I../common -o $@ $<
