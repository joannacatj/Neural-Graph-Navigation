# neugn_cuda

Pure C++17 + CUDA Runtime inference harness for exported NeuGN tensors.

Current supported scope:
- batch_size=1
- GCN encoder
- llama decoder
- fp32
- inference only

Run end-to-end:

```bash
bash scripts/run_cuda_compare.sh
```

This runs:
1. `method/export_cuda_params.py` to export weights, fixture inputs, and python reference outputs.
2. CMake build of `cuda_neugn`.
3. CUDA executable inference and output dump.
4. `scripts/compare_outputs.py` numeric comparison (atol/rtol = 1e-4 by default).

Run the C++/CUDA demo flow (similar purpose to `method/demo.py` for parity/timing on exported fixtures):

```bash
./build_cuda_neugn/neugn_demo \
  --export_dir ./cuda_export/hamster \
  --output ./cuda_export/hamster/cuda_output.bin \
  --python_ref ./cuda_export/hamster/python_output.bin \
  --shape ./cuda_export/hamster/python_output.shape \
  --warmup 1 \
  --runs 10 \
  --topk 5
```


Current forward implementation status:
- CUDA path now includes GCN encoder + llama decoder stack for batch_size=1 inference.
- Implementation uses naive custom CUDA kernels (no cuBLAS/cuDNN) and should be treated as functional reference, not optimized runtime.

## Fused GPU matcher (demo_cu)

`demo_cu` now runs filter-order-join on GPU with a fused DFSJoin+NeuGN kernel path.

Current limitations:
- batch query stream is supported, with one CUDA block per query.
- recommended query_size is <= 32.
- only GCN encoder + LLaMA decoder + fp32 export is supported.
- NeuGN only reorders local candidates and does not prune candidates.
