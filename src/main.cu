// Standalone nvcc build of the same kernels that bench.py runs through NVRTC.
//
//   nvcc -O3 -arch=native -o bench src/main.cu -lcublas
//   ./bench [gemm_size=2048] [log2_reduction_n=25]
//
// Checks each kernel for correctness and reports the median time over several runs,
// measured with cudaEvent timers.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "../kernels/reduction.cuh"
#include "../kernels/sgemm.cuh"

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),     \
                    __FILE__, __LINE__);                                              \
            exit(1);                                                                  \
        }                                                                             \
    } while (0)

#define CUBLAS_CHECK(call)                                                            \
    do {                                                                              \
        cublasStatus_t st_ = (call);                                                  \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                           \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)st_, __FILE__, __LINE__); \
            exit(1);                                                                  \
        }                                                                             \
    } while (0)

// Median time in ms of `reps` calls to f(), after one warm-up call.
template <typename F>
float time_ms(F f, int reps)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    f();
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> times(reps);
    for (int r = 0; r < reps; ++r) {
        CUDA_CHECK(cudaEventRecord(start));
        f();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&times[r], start, stop));
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::sort(times.begin(), times.end());
    return times[reps / 2];
}

typedef void (*ReduceKernel)(const float*, float*, unsigned int);

// Run a multi-pass reduction kernel until one value remains; returns the device pointer
// that holds the result.
static float* reduce_multipass(ReduceKernel k, int per_thread, const float* d_in,
                               unsigned int n, float* buf_a, float* buf_b)
{
    const float* src = d_in;
    float* dst = buf_a;
    while (n > 1) {
        unsigned int blocks = (n + RED_BLOCK * per_thread - 1) / (RED_BLOCK * per_thread);
        k<<<blocks, RED_BLOCK>>>(src, dst, n);
        n = blocks;
        src = dst;
        dst = (dst == buf_a) ? buf_b : buf_a;
    }
    return const_cast<float*>(src);
}

static void bench_reduction(unsigned int n, const cudaDeviceProp& prop, double peak_gbps)
{
    printf("\nReduction: sum of %u floats (%.0f MiB)\n", n, n * 4.0 / (1 << 20));
    std::vector<float> h(n);
    std::mt19937 gen(0);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    double ref = 0.0;
    for (unsigned int i = 0; i < n; ++i) { h[i] = dist(gen); ref += h[i]; }

    float *d_in, *buf_a, *buf_b, *d_out;
    unsigned int max_blocks = (n + RED_BLOCK - 1) / RED_BLOCK;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buf_a, max_blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buf_b, max_blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    struct { const char* name; ReduceKernel k; int per_thread; } kernels[] = {
        {"reduce0_interleaved_divergent", reduce0_interleaved_divergent, 1},
        {"reduce1_interleaved_bankconflict", reduce1_interleaved_bankconflict, 1},
        {"reduce2_sequential", reduce2_sequential, 1},
        {"reduce3_first_add", reduce3_first_add, 2},
        {"reduce4_warp_shuffle", reduce4_warp_shuffle, 2},
    };

    auto report = [&](const char* name, float ms, float result) {
        double gbps = n * 4.0 / (ms * 1e-3) / 1e9;
        double rel_err = std::fabs(result - ref) / ref;
        printf("  %-34s %8.3f ms  %7.1f GB/s (%5.1f%% peak)  err %.1e  %s\n", name, ms, gbps,
               100.0 * gbps / peak_gbps, rel_err, rel_err < 1e-4 ? "OK" : "WRONG");
    };

    for (auto& kc : kernels) {
        float* res = nullptr;
        float ms = time_ms([&] { res = reduce_multipass(kc.k, kc.per_thread, d_in, n, buf_a, buf_b); }, 50);
        float result;
        CUDA_CHECK(cudaMemcpy(&result, res, sizeof(float), cudaMemcpyDeviceToHost));
        report(kc.name, ms, result);
    }

    int blocks5 = prop.multiProcessorCount * (prop.maxThreadsPerMultiProcessor / RED_BLOCK);
    float ms = time_ms([&] {
        CUDA_CHECK(cudaMemsetAsync(d_out, 0, sizeof(float)));
        reduce5_gridstride_vec4<<<blocks5, RED_BLOCK>>>(d_in, d_out, n);
    }, 50);
    float result;
    CUDA_CHECK(cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    report("reduce5_gridstride_vec4", ms, result);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(buf_a));
    CUDA_CHECK(cudaFree(buf_b));
    CUDA_CHECK(cudaFree(d_out));
}

static void bench_sgemm(int s, double peak_gflops)
{
    printf("\nSGEMM: %dx%d FP32, C = A @ B\n", s, s);
    if (s % 128 != 0) { fprintf(stderr, "size must be a multiple of 128\n"); exit(1); }
    const int M = s, N = s, K = s;
    size_t bytes = (size_t)s * s * sizeof(float);
    std::vector<float> hA((size_t)s * s), hB((size_t)s * s), hC((size_t)s * s), hRef((size_t)s * s);
    std::mt19937 gen(s);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : hA) x = dist(gen);
    for (auto& x : hB) x = dist(gen);

    float *A, *B, *C;
    CUDA_CHECK(cudaMalloc(&A, bytes));
    CUDA_CHECK(cudaMalloc(&B, bytes));
    CUDA_CHECK(cudaMalloc(&C, bytes));
    CUDA_CHECK(cudaMemcpy(A, hA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B, hB.data(), bytes, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));   // no TF32
    const float alpha = 1.0f, beta = 0.0f;
    // cuBLAS is column-major: computing C^T = B^T A^T gives row-major C = A B.
    auto cublas = [&] {
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K,
                                 &beta, C, N));
    };
    CUDA_CHECK(cudaMemset(C, 0, bytes));
    cublas();
    CUDA_CHECK(cudaMemcpy(hRef.data(), C, bytes, cudaMemcpyDeviceToHost));
    float scale = 0.0f;
    for (float x : hRef) scale = std::max(scale, std::fabs(x));

    const double flops = 2.0 * M * N * K;
    int reps = std::max(3, (int)(20.0 * std::pow(1024.0 / s, 3)));
    auto report = [&](const char* name, float ms) {
        CUDA_CHECK(cudaMemcpy(hC.data(), C, bytes, cudaMemcpyDeviceToHost));
        float err = 0.0f;
        for (size_t i = 0; i < hC.size(); ++i) err = std::max(err, std::fabs(hC[i] - hRef[i]));
        err /= scale;
        double gflops = flops / (ms * 1e-3) / 1e9;
        printf("  %-22s %9.3f ms  %8.1f GFLOP/s (%5.1f%% peak)  err %.1e  %s\n", name, ms,
               gflops, 100.0 * gflops / peak_gflops, err, err < 1e-4 ? "OK" : "WRONG");
    };

    dim3 b32(32, 32);
    dim3 g_naive((M + 31) / 32, (N + 31) / 32), g_tile((N + 31) / 32, (M + 31) / 32);
    dim3 g1(N / BN1, M / BM1), g2(N / BN2, M / BM2);

    struct { const char* name; dim3 grid, block; } cfg[] = {
        {"sgemm0_naive", g_naive, b32},
        {"sgemm1_coalesced", g_tile, b32},
        {"sgemm2_smem_tiled", g_tile, b32},
        {"sgemm3_1d_blocktile", g1, dim3((BM1 * BN1) / TM1)},
        {"sgemm4_2d_blocktile", g2, dim3((BM2 * BN2) / (TM2 * TN2))},
        {"sgemm5_vectorized", g2, dim3((BM2 * BN2) / (TM2 * TN2))},
    };
    typedef void (*GemmKernel)(int, int, int, float, const float*, const float*, float, float*);
    GemmKernel fns[] = {sgemm0_naive, sgemm1_coalesced, sgemm2_smem_tiled,
                        sgemm3_1d_blocktile, sgemm4_2d_blocktile, sgemm5_vectorized};

    for (int i = 0; i < 6; ++i) {
        CUDA_CHECK(cudaMemset(C, 0, bytes));
        float ms = time_ms([&] { fns[i]<<<cfg[i].grid, cfg[i].block>>>(M, N, K, alpha, A, B, beta, C); },
                           i == 0 ? 3 : reps);
        report(cfg[i].name, ms);
    }
    report("cublas", time_ms(cublas, reps));

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));
}

int main(int argc, char** argv)
{
    int gemm_size = argc > 1 ? atoi(argv[1]) : 2048;
    int log2_n = argc > 2 ? atoi(argv[2]) : 25;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int clock_khz = 0, mem_clock_khz = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
    CUDA_CHECK(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, 0));
    // FP32 lanes per SM: 128 on sm_86/89, 64 on sm_70/75/80. Used only for the % of peak column.
    int lanes = (prop.major == 8 && prop.minor >= 6) || prop.major >= 9 ? 128 : 64;
    double peak_gflops = prop.multiProcessorCount * lanes * 2.0 * clock_khz / 1e6;
    double peak_gbps = 2.0 * mem_clock_khz * 1e3 * (prop.memoryBusWidth / 8) / 1e9;
    printf("%s  sm_%d%d  %d SMs  peak %.0f GFLOP/s FP32, %.0f GB/s DRAM\n", prop.name,
           prop.major, prop.minor, prop.multiProcessorCount, peak_gflops, peak_gbps);

    bench_reduction(1u << log2_n, prop, peak_gbps);
    bench_sgemm(gemm_size, peak_gflops);
    return 0;
}
