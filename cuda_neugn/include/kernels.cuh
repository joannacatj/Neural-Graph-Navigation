#pragma once

#include <cuda_runtime.h>

// NOTE: current CUDA implementation supports only:
// - batch_size=1
// - GCN encoder
// - llama decoder
// - fp32
// - inference only

void launch_copy_kernel(const float* in, float* out, int n, cudaStream_t stream = 0);
void launch_linear_kernel(const float* x, const float* w, const float* b, float* y, int rows, int in_dim, int out_dim, cudaStream_t stream = 0);
void launch_gelu_kernel(float* x, int n, cudaStream_t stream = 0);
