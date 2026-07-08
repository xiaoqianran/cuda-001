NVCC?=nvcc
all: terrain
terrain: src/terrain.cu
	$(NVCC) -arch=sm_75 -O3 -I../common -o $@ $<
