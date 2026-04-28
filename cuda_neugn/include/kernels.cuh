#pragma once

#include <cuda_runtime.h>

// NOTE: current CUDA implementation supports only:
// - batch_size=1
// - GCN encoder
// - llama decoder
// - fp32
// - inference only

void launch_copy_kernel(const float* in, float* out, int n, cudaStream_t stream = 0);
