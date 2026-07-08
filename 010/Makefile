NVCC?=nvcc
all: scene_rt
scene_rt: src/scene_rt.cu
	$(NVCC) -arch=sm_75 -O3 -I../common -o $@ $<
clean: ; rm -f scene_rt
