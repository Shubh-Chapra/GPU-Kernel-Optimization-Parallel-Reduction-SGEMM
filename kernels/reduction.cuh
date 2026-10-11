// Parallel sum reduction: six kernels, each fixing one bottleneck of the previous.
// Device code only, so the same source compiles with nvcc (src/main.cu) and with
// NVRTC at runtime (bench.py). Kernels are extern "C" so NVRTC can look them up by name.
//
// reduce0..reduce4 write one partial sum per block; the host relaunches the kernel on
// the partials until one value is left. reduce5 finishes in a single launch with atomics.

#ifndef REDUCTION_CUH
#define REDUCTION_CUH

#define RED_BLOCK 256   // threads per block for every reduction kernel

// v0: interleaved addressing with a modulo test.
// Threads in the same warp take different branches (tid % (2*s)), so most of each warp
// idles and the modulo itself is expensive.
extern "C" __global__ void reduce0_interleaved_divergent(const float* in, float* out, unsigned int n)
{
    __shared__ float sdata[RED_BLOCK];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// v1: interleaved addressing with contiguous active threads.
// Removes the divergence, but the strided index 2*s*tid makes threads in a warp hit
// the same shared-memory bank, so accesses serialize (bank conflicts).
extern "C" __global__ void reduce1_interleaved_bankconflict(const float* in, float* out, unsigned int n)
{
    __shared__ float sdata[RED_BLOCK];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        unsigned int index = 2 * s * tid;
        if (index < blockDim.x)
            sdata[index] += sdata[index + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// v2: sequential addressing. Active threads read consecutive words, so there are no
// bank conflicts. Half the threads still sit idle after the very first step.
extern "C" __global__ void reduce2_sequential(const float* in, float* out, unsigned int n)
{
    __shared__ float sdata[RED_BLOCK];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// v3: first add during the global load. Each thread loads two elements, which halves
// the number of blocks and puts every thread to work on the first addition.
extern "C" __global__ void reduce3_first_add(const float* in, float* out, unsigned int n)
{
    __shared__ float sdata[RED_BLOCK];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    float sum = (i < n) ? in[i] : 0.0f;
    if (i + blockDim.x < n) sum += in[i + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// v4: v3 plus a warp-shuffle finish. Once 32 values remain they live in one warp, so
// __shfl_down_sync exchanges registers directly: no shared memory, no __syncthreads.
extern "C" __global__ void reduce4_warp_shuffle(const float* in, float* out, unsigned int n)
{
    __shared__ float sdata[RED_BLOCK];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    float sum = (i < n) ? in[i] : 0.0f;
    if (i + blockDim.x < n) sum += in[i + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s)
            sdata[tid] = sum = sum + sdata[tid + s];
        __syncthreads();
    }
    if (tid < 32) {
        sum += sdata[tid + 32];
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (tid == 0) out[blockIdx.x] = sum;
}

// v5: grid-stride loop with float4 loads, then a shuffle-only block reduction and one
// atomicAdd per block. The grid is sized to just fill the GPU, so each thread sums many
// elements in registers and the whole reduction is one launch. *out must be zeroed first.
extern "C" __global__ void reduce5_gridstride_vec4(const float* in, float* out, unsigned int n)
{
    __shared__ float warp_sums[RED_BLOCK / 32];
    unsigned int tid = threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    float sum = 0.0f;

    unsigned int n4 = n / 4;
    const float4* in4 = reinterpret_cast<const float4*>(in);
    for (unsigned int i = blockIdx.x * blockDim.x + tid; i < n4; i += stride) {
        float4 v = in4[i];
        sum += (v.x + v.y) + (v.z + v.w);
    }
    for (unsigned int i = n4 * 4 + blockIdx.x * blockDim.x + tid; i < n; i += stride)
        sum += in[i];   // the 0-3 elements left over when n is not a multiple of 4

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    unsigned int lane = tid & 31, warp = tid >> 5;
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        sum = (lane < blockDim.x / 32) ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (lane == 0) atomicAdd(out, sum);
    }
}

#endif
