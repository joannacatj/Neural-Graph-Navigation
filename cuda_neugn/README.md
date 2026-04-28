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


Current forward implementation status:
- CUDA computes decoder output head (`decoder.output`) from exported `python_graph_features` as validation target.
- Full end-to-end transformer stack CUDA parity is still in progress.
