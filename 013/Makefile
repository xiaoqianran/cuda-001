NVCC?=nvcc
all: materials
materials: src/materials.cu
	$(NVCC) -arch=sm_75 -O3 -I../common -o $@ $<
