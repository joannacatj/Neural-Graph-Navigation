#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${CONFIG_PATH:-./method/model_params/wikics}
GRAPH_PATH=${GRAPH_PATH:-./datasets/wikics}
DATASET=${DATASET:-wikics}
OUT_DIR=${OUT_DIR:-./cuda_export/wikics}
DEVICE=${DEVICE:-cuda}

pushd method >/dev/null
python export_cuda_params.py \
  --config_path "${CONFIG_PATH#./method/}" \
  --graph_path "../${GRAPH_PATH#./}" \
  --dataset "$DATASET" \
  --out_dir "../${OUT_DIR#./}" \
  --device "$DEVICE" \
  --seed 42 \
  --query_size 20
popd >/dev/null

cmake -S cuda_neugn -B build_cuda_neugn -DCMAKE_BUILD_TYPE=Release
cmake --build build_cuda_neugn -j

./build_cuda_neugn/neugn_cuda \
  --export_dir "$OUT_DIR" \
  --output "$OUT_DIR/cuda_output.bin" \
  --mode full \
  --print_first 10

python scripts/compare_outputs.py \
  --a "$OUT_DIR/python_output.bin" \
  --b "$OUT_DIR/cuda_output.bin" \
  --shape "$OUT_DIR/python_output.shape" \
  --atol 1e-4 \
  --rtol 1e-4
