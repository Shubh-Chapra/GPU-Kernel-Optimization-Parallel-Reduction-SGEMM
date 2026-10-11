// SGEMM, C = alpha * A * B + beta * C, row-major, A is MxK, B is KxN, C is MxN.
// Six kernels, each fixing the main bottleneck of the previous one.
// Device code only, shared by src/main.cu (nvcc) and bench.py (NVRTC).
//
// sgemm0..sgemm2 handle any size. sgemm3 needs M, N multiples of 64 and K a multiple of 8;
// sgemm4 and sgemm5 need M, N multiples of 128 and K a multiple of 8.

#ifndef SGEMM_CUH
#define SGEMM_CUH

#define TILE 32         // block edge for sgemm0..sgemm2 (32x32 = 1024 threads)

// sgemm3: 1D block tiling
#define BM1 64
#define BN1 64
#define BK1 8
#define TM1 8           // outputs per thread; BM1*BN1/TM1 = 512 threads per block

// sgemm4 / sgemm5: 2D block tiling
#define BM2 128
#define BN2 128
#define BK2 8
#define TM2 8
#define TN2 8           // 8x8 outputs per thread; BM2*BN2/(TM2*TN2) = 256 threads per block

// v0: one thread per output, but threadIdx.x walks down rows. Consecutive threads in a
// warp read A from different rows (K floats apart) and write C K floats apart, so global
// memory accesses are not coalesced.
extern "C" __global__ void sgemm0_naive(int M, int N, int K, float alpha, const float* A,
                                        const float* B, float beta, float* C)
{
    int row = blockIdx.x * TILE + threadIdx.x;
    int col = blockIdx.y * TILE + threadIdx.y;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k)
            acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = alpha * acc + beta * C[row * N + col];
    }
}

// v1: same math, but threadIdx.x walks along columns. A warp now reads 32 consecutive
// floats of B and writes 32 consecutive floats of C (coalesced), and reads one broadcast
// value of A.
extern "C" __global__ void sgemm1_coalesced(int M, int N, int K, float alpha, const float* A,
                                            const float* B, float beta, float* C)
{
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k)
            acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = alpha * acc + beta * C[row * N + col];
    }
}

// v2: shared-memory tiling. The block stages a 32x32 tile of A and of B in shared memory,
// so each global element is loaded once per block instead of once per thread (32x reuse).
extern "C" __global__ void sgemm2_smem_tiled(int M, int N, int K, float alpha, const float* A,
                                             const float* B, float beta, float* C)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;
    for (int t = 0; t < K; t += TILE) {
        As[ty][tx] = (row < M && t + tx < K) ? A[row * K + t + tx] : 0.0f;
        Bs[ty][tx] = (t + ty < K && col < N) ? B[(t + ty) * N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE; ++k)
            acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < M && col < N)
        C[row * N + col] = alpha * acc + beta * C[row * N + col];
}

// v3: 1D block tiling. Each thread computes TM1 outputs in one column. The B value it reads
// from shared memory is reused TM1 times from a register, so the kernel does more math
// per shared-memory load (v2 does 2 shared loads per FMA).
extern "C" __global__ void sgemm3_1d_blocktile(int M, int N, int K, float alpha, const float* A,
                                               const float* B, float beta, float* C)
{
    __shared__ float As[BM1 * BK1];
    __shared__ float Bs[BK1 * BN1];
    int tid = threadIdx.x;
    int threadCol = tid % BN1;
    int threadRow = tid / BN1;            // 0 .. BM1/TM1 - 1

    A += blockIdx.y * BM1 * K;
    B += blockIdx.x * BN1;
    C += blockIdx.y * BM1 * N + blockIdx.x * BN1;

    // 512 threads load the 64x8 A tile and the 8x64 B tile, one element each.
    int innerRowA = tid / BK1, innerColA = tid % BK1;
    int innerRowB = tid / BN1, innerColB = tid % BN1;

    float acc[TM1] = {0.0f};
    for (int bk = 0; bk < K; bk += BK1) {
        As[innerRowA * BK1 + innerColA] = A[innerRowA * K + innerColA];
        Bs[innerRowB * BN1 + innerColB] = B[innerRowB * N + innerColB];
        __syncthreads();
        A += BK1;
        B += BK1 * N;

        for (int d = 0; d < BK1; ++d) {
            float b = Bs[d * BN1 + threadCol];
            for (int r = 0; r < TM1; ++r)
                acc[r] += As[(threadRow * TM1 + r) * BK1 + d] * b;
        }
        __syncthreads();
    }
    for (int r = 0; r < TM1; ++r) {
        int idx = (threadRow * TM1 + r) * N + threadCol;
        C[idx] = alpha * acc[r] + beta * C[idx];
    }
}

// v4: 2D block tiling. Each thread computes an 8x8 output tile as an outer product of 8
// values of A and 8 values of B held in registers: 16 shared loads feed 64 FMAs.
extern "C" __global__ void sgemm4_2d_blocktile(int M, int N, int K, float alpha, const float* A,
                                               const float* B, float beta, float* C)
{
    __shared__ float As[BM2 * BK2];
    __shared__ float Bs[BK2 * BN2];
    const int numThreads = (BM2 * BN2) / (TM2 * TN2);     // 256
    int tid = threadIdx.x;
    int threadCol = tid % (BN2 / TN2);
    int threadRow = tid / (BN2 / TN2);

    A += blockIdx.y * BM2 * K;
    B += blockIdx.x * BN2;
    C += blockIdx.y * BM2 * N + blockIdx.x * BN2;

    // Each thread loads 4 elements of the 128x8 A tile and 4 of the 8x128 B tile.
    int innerRowA = tid / BK2, innerColA = tid % BK2;
    const int strideA = numThreads / BK2;                 // 32 rows per pass
    int innerRowB = tid / BN2, innerColB = tid % BN2;
    const int strideB = numThreads / BN2;                 // 2 rows per pass

    float acc[TM2 * TN2] = {0.0f};
    float regM[TM2], regN[TN2];

    for (int bk = 0; bk < K; bk += BK2) {
        for (int off = 0; off < BM2; off += strideA)
            As[(innerRowA + off) * BK2 + innerColA] = A[(innerRowA + off) * K + innerColA];
        for (int off = 0; off < BK2; off += strideB)
            Bs[(innerRowB + off) * BN2 + innerColB] = B[(innerRowB + off) * N + innerColB];
        __syncthreads();
        A += BK2;
        B += BK2 * N;

        for (int d = 0; d < BK2; ++d) {
            for (int i = 0; i < TM2; ++i) regM[i] = As[(threadRow * TM2 + i) * BK2 + d];
            for (int j = 0; j < TN2; ++j) regN[j] = Bs[d * BN2 + threadCol * TN2 + j];
            for (int i = 0; i < TM2; ++i)
                for (int j = 0; j < TN2; ++j)
                    acc[i * TN2 + j] += regM[i] * regN[j];
        }
        __syncthreads();
    }
    for (int i = 0; i < TM2; ++i)
        for (int j = 0; j < TN2; ++j) {
            int idx = (threadRow * TM2 + i) * N + threadCol * TN2 + j;
            C[idx] = alpha * acc[i * TN2 + j] + beta * C[idx];
        }
}

// v5: v4 with 128-bit (float4) global loads and stores, and the A tile stored transposed
// in shared memory (As[k][m]) so the inner loop reads A along contiguous addresses too.
extern "C" __global__ void sgemm5_vectorized(int M, int N, int K, float alpha, const float* A,
                                             const float* B, float beta, float* C)
{
    __shared__ float As[BK2 * BM2];      // transposed: As[d * BM2 + m]
    __shared__ float Bs[BK2 * BN2];
    int tid = threadIdx.x;
    int threadCol = tid % (BN2 / TN2);
    int threadRow = tid / (BN2 / TN2);

    A += blockIdx.y * BM2 * K;
    B += blockIdx.x * BN2;
    C += blockIdx.y * BM2 * N + blockIdx.x * BN2;

    // 256 threads, one float4 of A (128x8 tile) and one float4 of B (8x128 tile) each.
    int innerRowA = tid / (BK2 / 4), innerColA = tid % (BK2 / 4);
    int innerRowB = tid / (BN2 / 4), innerColB = tid % (BN2 / 4);

    float acc[TM2 * TN2] = {0.0f};
    float regM[TM2], regN[TN2];

    for (int bk = 0; bk < K; bk += BK2) {
        float4 a = reinterpret_cast<const float4*>(&A[innerRowA * K + innerColA * 4])[0];
        As[(innerColA * 4 + 0) * BM2 + innerRowA] = a.x;
        As[(innerColA * 4 + 1) * BM2 + innerRowA] = a.y;
        As[(innerColA * 4 + 2) * BM2 + innerRowA] = a.z;
        As[(innerColA * 4 + 3) * BM2 + innerRowA] = a.w;
        reinterpret_cast<float4*>(&Bs[innerRowB * BN2 + innerColB * 4])[0] =
            reinterpret_cast<const float4*>(&B[innerRowB * N + innerColB * 4])[0];
        __syncthreads();
        A += BK2;
        B += BK2 * N;

        for (int d = 0; d < BK2; ++d) {
            for (int i = 0; i < TM2; ++i) regM[i] = As[d * BM2 + threadRow * TM2 + i];
            for (int j = 0; j < TN2; ++j) regN[j] = Bs[d * BN2 + threadCol * TN2 + j];
            for (int i = 0; i < TM2; ++i)
                for (int j = 0; j < TN2; ++j)
                    acc[i * TN2 + j] += regM[i] * regN[j];
        }
        __syncthreads();
    }
    for (int i = 0; i < TM2; ++i)
        for (int j = 0; j < TN2; j += 4) {
            float4* cp = reinterpret_cast<float4*>(&C[(threadRow * TM2 + i) * N + threadCol * TN2 + j]);
            float4 c = cp[0];
            c.x = alpha * acc[i * TN2 + j + 0] + beta * c.x;
            c.y = alpha * acc[i * TN2 + j + 1] + beta * c.y;
            c.z = alpha * acc[i * TN2 + j + 2] + beta * c.z;
            c.w = alpha * acc[i * TN2 + j + 3] + beta * c.w;
            cp[0] = c;
        }
}

#endif
