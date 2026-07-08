NVCC?=nvcc
all: pbr
pbr: src/pbr.cu
	$(NVCC) -arch=sm_75 -O3 -I../common -o $@ $<
