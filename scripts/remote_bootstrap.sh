#!/usr/bin/env bash
# Bootstrap a fresh Lambda GPU instance (Ubuntu + CUDA) for this experiment.
# Usage (from the instance, inside the synced repo dir):  bash scripts/remote_bootstrap.sh
set -euo pipefail

python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu124 || \
  .venv/bin/pip install -q torch
.venv/bin/pip install -q "transformers>=4.51" datasets accelerate pyyaml matplotlib \
  pandas scipy math-verify pytest "git+https://github.com/anthropics/jacobian-lens.git"

# Prefetch model, lens, and datasets so runs don't stall on downloads.
.venv/bin/python - <<'EOF'
from huggingface_hub import snapshot_download, hf_hub_download
snapshot_download("Qwen/Qwen3-4B")
hf_hub_download("neuronpedia/jacobian-lens",
                "qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt")
from datasets import load_dataset
load_dataset("openai/gsm8k", "main", split="test")
load_dataset("HuggingFaceH4/MATH-500", split="test")
load_dataset("HuggingFaceH4/aime_2024", split="train")
load_dataset("stanfordnlp/sst2", split="validation")
print("prefetch done")
EOF

.venv/bin/python -m pytest tests -q
echo "bootstrap complete"
