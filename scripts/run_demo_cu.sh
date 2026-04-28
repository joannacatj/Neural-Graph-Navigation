#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${CONFIG_PATH:-./method/model_params/wikics}
GRAPH_PATH=${GRAPH_PATH:-./datasets/wikics}
DATASET=${DATASET:-wikics}
OUT_DIR=${OUT_DIR:-./cuda_export/wikics}
DEVICE=${DEVICE:-cuda}

pushd method >/dev/null
python export_demo_inputs.py \
  --config_path "${CONFIG_PATH#./method/}" \
  --graph_path "../${GRAPH_PATH#./}" \
  --dataset "$DATASET" \
  --out_dir "../${OUT_DIR#./}" \
  --query_size 20 \
  --num_queries 200 \
  --nav_depth 10 \
  --seed 42 \
  --device "$DEVICE" \
  --output_python_csv "../${OUT_DIR#./}/demo_py_results.csv"
popd >/dev/null

cmake -S cuda_neugn -B build_cuda_neugn -DCMAKE_BUILD_TYPE=Release
cmake --build build_cuda_neugn -j --target demo_cu

./build_cuda_neugn/demo_cu \
  --export_dir "$OUT_DIR" \
  --query_bin "$OUT_DIR/demo_input/queries.bin" \
  --output "$OUT_DIR/demo_cu_results.csv" \
  --query_size 20 \
  --num_queries 200 \
  --nav_depth 10 \
  --seed 42 \
  --print_first 5

python scripts/compare_demo_py_cu.py \
  --python_csv "$OUT_DIR/demo_py_results.csv" \
  --cuda_csv "$OUT_DIR/demo_cu_results.csv"
