NVCC?=nvcc
all: path_gi
path_gi: src/path_gi.cu
	$(NVCC) -arch=sm_75 -O3 -I../common -o $@ $<
