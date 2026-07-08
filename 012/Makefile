NVCC?=nvcc
all: aa_dof
aa_dof: src/aa_dof.cu
	$(NVCC) -arch=sm_75 -O3 -I../common -o $@ $<
