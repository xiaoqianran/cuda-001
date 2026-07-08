NVCC?=nvcc
all: mesh_bvh
mesh_bvh: src/mesh_bvh.cu
	$(NVCC) -arch=sm_75 -O3 -I../common -o $@ $<
clean:; rm -f mesh_bvh
