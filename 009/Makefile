NVCC?=nvcc
all: simple_rt
simple_rt: src/simple_rt.cu
	$(NVCC) -arch=sm_75 -O3 -I../common -o $@ $<
clean: ; rm -f simple_rt
