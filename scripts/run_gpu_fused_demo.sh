#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-./cuda_export/hamster}"
DATASET="${2:-hamster}"

python method/export_demo_inputs.py \
  --export_dir "$OUT_DIR" \
  --dataset "$DATASET"

cmake -S cuda_neugn -B build_cuda_neugn -DCMAKE_BUILD_TYPE=Release
cmake --build build_cuda_neugn -j --target demo_cu

./build_cuda_neugn/demo_cu \
  --export_dir "$OUT_DIR" \
  --query_bin "$OUT_DIR/demo_input/queries.bin" \
  --output "$OUT_DIR/demo_cu_fused_results.csv" \
  --query_size 20 \
  --num_queries 200 \
  --nav_depth 10 \
  --seed 42 \
  --print_first 5

if [[ -f "$OUT_DIR/demo_results.csv" && -f scripts/compare_demo_py_cu.py ]]; then
  python scripts/compare_demo_py_cu.py \
    --py_csv "$OUT_DIR/demo_results.csv" \
    --cu_csv "$OUT_DIR/demo_cu_fused_results.csv"
fi
