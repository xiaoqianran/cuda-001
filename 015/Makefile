NVCC?=nvcc
all: rt_opt
rt_opt: src/rt_opt.cu
	$(NVCC) -arch=sm_75 -O3 -I../common -o $@ $<
