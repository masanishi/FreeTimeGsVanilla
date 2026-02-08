uv venv --python 3.12.12 .venv

source .venv/bin/activate

uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

uv pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu128.html

uv sync --no-build-isolation

uv pip install romatch[fused-local-corr] rich