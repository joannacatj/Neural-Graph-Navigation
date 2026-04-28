#include "kernels.cuh"

__global__ void copy_kernel(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx];
}

void launch_copy_kernel(const float* in, float* out, int n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    copy_kernel<<<grid, block, 0, stream>>>(in, out, n);
}
