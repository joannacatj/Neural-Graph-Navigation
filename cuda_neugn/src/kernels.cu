#include "kernels.cuh"

#include <math.h>

__global__ void copy_kernel(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx];
}

__global__ void linear_kernel(const float* x, const float* w, const float* b, float* y, int rows, int in_dim, int out_dim) {
    int r = blockIdx.y;
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows && o < out_dim) {
        float acc = b ? b[o] : 0.0f;
        for (int i = 0; i < in_dim; ++i) {
            acc += x[r * in_dim + i] * w[o * in_dim + i];
        }
        y[r * out_dim + o] = acc;
    }
}

__global__ void gelu_kernel(float* x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = x[idx];
        float c = 0.7978845608f * (v + 0.044715f * v * v * v);
        x[idx] = 0.5f * v * (1.0f + tanhf(c));
    }
}

void launch_copy_kernel(const float* in, float* out, int n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    copy_kernel<<<grid, block, 0, stream>>>(in, out, n);
}

void launch_linear_kernel(const float* x, const float* w, const float* b, float* y, int rows, int in_dim, int out_dim, cudaStream_t stream) {
    int block = 256;
    int gx = (out_dim + block - 1) / block;
    dim3 grid(gx, rows, 1);
    linear_kernel<<<grid, block, 0, stream>>>(x, w, b, y, rows, in_dim, out_dim);
}

void launch_gelu_kernel(float* x, int n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    gelu_kernel<<<grid, block, 0, stream>>>(x, n);
}
