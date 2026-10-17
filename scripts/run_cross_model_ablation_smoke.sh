#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-/content/txnagentbench}"
ARTIFACT_ROOT="${2:-/content/drive/MyDrive/txnagentbench-artifacts}"
OUTPUT_DIR="$ARTIFACT_ROOT/cross-model-ablation-smoke-v1"

cd "$PROJECT_DIR"

if [[ ! -d "$ARTIFACT_ROOT" ]]; then
  echo "Persistent artifact root is missing: $ARTIFACT_ROOT" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "The Git worktree is dirty. Model experiments require a clean commit." >&2
  exit 2
fi

python - <<'PY'
import torch

assert torch.cuda.is_available(), "A CUDA GPU runtime is required."
print("GPU:", torch.cuda.get_device_name(0))
print("GPU memory GiB:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
PY

python -m pip install -e ".[gpu,analysis,test]"
python -m pytest -q

if [[ -f "$OUTPUT_DIR/artifact_manifest.json" ]]; then
  txnagentbench verify-manifest --output "$OUTPUT_DIR"
  echo "Checkpoint already completed and verified."
  exit 0
fi

txnagentbench plan --config configs/cross_model_ablation_smoke.json
txnagentbench run \
  --config configs/cross_model_ablation_smoke.json \
  --output "$OUTPUT_DIR"

python - "$OUTPUT_DIR" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
summaries = [
    json.loads(path.read_text())
    for path in sorted((root / "runs").glob("*/summary.json"))
]
assert len(summaries) == 200, f"Expected 200 runs, found {len(summaries)}"
assert sum(row["invalid_actions"] for row in summaries) == 0

by_policy = defaultdict(list)
for row in summaries:
    by_policy[row["policy"]].append(row)

for policy in ["serial_recovery", "occ_recovery", "txnguard"]:
    assert len(by_policy[policy]) == 40
    assert all(row["task_success"] for row in by_policy[policy]), policy
    assert all(not row["severe_violation"] for row in by_policy[policy]), policy

assert all(
    row["task_success"]
    == (row["contention"] == 0 or row["fault_mode"] == "compensation_failure")
    for row in by_policy["recovery_only"]
)
assert all(
    row["task_success"] == (row["fault_mode"] == "none")
    for row in by_policy["txnguard_no_recovery"]
)

models = sorted({row["model_name"] for row in summaries})
assert models == ["mistral-7b-instruct-v0.3", "qwen2.5-7b-instruct"]
print("PASS: 200-run cross-model recovery ablation gate")
print("Models:", models)
print("Runs:", len(summaries))
print("Invalid actions:", sum(row["invalid_actions"] for row in summaries))
PY

bash scripts/finalize_artifact.sh "$OUTPUT_DIR"
