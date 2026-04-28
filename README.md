# Neural-Graph-Navigation
the code of the paper : Neural Graph Navigation for Efficient Subgraph Matching


# **Overview**

```
method/
├── README.md
├── NeuGN/
│   ├── encoders/
│   │   ├── mpnn_encoder.py (GCN, GIN)
│   │   ├── nag_encoder.py (NAGphormer)
│   ├── graph_tokenizer.py (as the name suggests)
│   ├── datasets.py (process for datasets)
│   ├── nx_utils.py (contains a series of NetworkX utilities for obtaining Eulerian paths)
│   ├── utils.py (contains some data processing functions)
│   ├── model.py (NeuGN related)
├── main.py
```

Subgraph Matching Dataset: See ./datasets/subgraphmatching

## **Training Setup**

- Create the necessary folders and set the parameters (see ./data_params/wikics/model_args.yaml for details).
- Just run the code.

```bash
Example:
python -m torch.distributed.launch --nproc_per_node 4 main.py --config ./model_params/wikics--load_params 0
```

## CUDA export + output comparison quickstart

Use the helper script below from the repository root to run the end-to-end CUDA parity workflow (export fixtures, build `cuda_neugn`, run CUDA inference, compare against Python output):

```bash
bash scripts/run_cuda_compare.sh
```

You can override dataset-specific paths with environment variables:

```bash
CONFIG_PATH=./method/model_params/hamster \
GRAPH_PATH=./datasets/hamster \
DATASET=hamster \
OUT_DIR=./cuda_export/hamster \
DEVICE=cuda \
bash scripts/run_cuda_compare.sh
```

Manual equivalent (for debugging each stage) is:

```bash
cd method
python export_cuda_params.py \
  --config_path ./model_params/wikics \
  --graph_path ../datasets/wikics \
  --dataset wikics \
  --out_dir ../cuda_export/wikics \
  --device cuda \
  --seed 42 \
  --query_size 20
cd ..

cmake -S cuda_neugn -B build_cuda_neugn -DCMAKE_BUILD_TYPE=Release
cmake --build build_cuda_neugn -j

./build_cuda_neugn/neugn_cuda \
  --export_dir ./cuda_export/wikics \
  --output ./cuda_export/wikics/cuda_output.bin \
  --mode full \
  --print_first 10

python scripts/compare_outputs.py \
  --a ./cuda_export/wikics/python_output.bin \
  --b ./cuda_export/wikics/cuda_output.bin \
  --shape ./cuda_export/wikics/python_output.shape \
  --atol 1e-4 \
  --rtol 1e-4
```
