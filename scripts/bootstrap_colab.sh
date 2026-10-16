#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-/content/txnagentbench}"
cd "$PROJECT_DIR"

python -m pip install --upgrade pip
python -m pip install -e ".[gpu,analysis,test]"

mkdir -p /content/txnagentbench-artifacts

python - <<'PY'
import platform
print("Python environment ready")
print(platform.platform())
try:
    import torch
    print("torch", torch.__version__)
    print("cuda_available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu", torch.cuda.get_device_name(0))
        print("gpu_memory_gib", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
except Exception as exc:
    print("torch_check_failed", repr(exc))
PY

txnagentbench plan --config configs/smoke.json

