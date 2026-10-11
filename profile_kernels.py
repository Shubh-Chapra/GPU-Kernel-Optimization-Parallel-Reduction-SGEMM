"""Launch each kernel once, so Nsight Compute can profile it.

Run under ncu (see run_ncu.ps1); on its own it only checks that every launch works.
Sizes match the benchmark: 2^25-element reduction, 2048^3 SGEMM.
"""
import time

import cupy as cp
import numpy as np

from bench import RED_BLOCK, REDUCTION_KERNELS, SGEMM_KERNELS, load_module, sgemm_launch

N_RED = 1 << 25
GEMM = 2048


def warm_up(A, B, seconds=0.5):
    """Keep the GPU busy briefly so a laptop GPU is at full clocks when the kernel runs.
    cuBLAS kernels don't match ncu's kernel filter, so they are not profiled."""
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        A @ B
        cp.cuda.Device().synchronize()


def main():
    A = cp.random.rand(GEMM, GEMM, dtype=cp.float32)
    B = cp.random.rand(GEMM, GEMM, dtype=cp.float32)
    C = cp.zeros((GEMM, GEMM), dtype=cp.float32)

    red = load_module("reduction.cuh")
    x = cp.random.rand(N_RED, dtype=cp.float32)
    for name, per_thread in REDUCTION_KERNELS:
        blocks = (N_RED + RED_BLOCK * per_thread - 1) // (RED_BLOCK * per_thread)
        out = cp.empty(blocks, dtype=cp.float32)
        warm_up(A, B)
        red.get_function(name)((blocks,), (RED_BLOCK,), (x, out, np.uint32(N_RED)))   # first pass only
    sms = cp.cuda.runtime.getDeviceProperties(0)["multiProcessorCount"]
    out = cp.zeros(1, dtype=cp.float32)
    warm_up(A, B)
    red.get_function("reduce5_gridstride_vec4")((sms * 6,), (RED_BLOCK,), (x, out, np.uint32(N_RED)))

    gemm = load_module("sgemm.cuh")
    for name in SGEMM_KERNELS:
        grid, block = sgemm_launch(name, GEMM, GEMM, GEMM)
        warm_up(A, B)
        gemm.get_function(name)(grid, block, (np.int32(GEMM), np.int32(GEMM), np.int32(GEMM),
                                              np.float32(1.0), A, B, np.float32(0.0), C))
    cp.cuda.Device().synchronize()
    print("launched all kernels")


if __name__ == "__main__":
    main()
