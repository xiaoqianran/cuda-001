NVCC?=nvcc
all: volume_mb
volume_mb: src/volume_mb.cu
	$(NVCC) -arch=sm_75 -O3 -I../common -o $@ $<
