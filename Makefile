NVCC  ?= nvcc
ARCH  ?= native
FLAGS  = -O3 -std=c++17 -arch=$(ARCH) -lineinfo

bench: src/main.cu kernels/reduction.cuh kernels/sgemm.cuh
	$(NVCC) $(FLAGS) -o $@ src/main.cu -lcublas

# Register and shared-memory usage per kernel, as reported by ptxas.
resources: src/main.cu kernels/reduction.cuh kernels/sgemm.cuh
	$(NVCC) $(FLAGS) -Xptxas -v -c -o /dev/null src/main.cu

clean:
	rm -f bench

.PHONY: resources clean
