#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-/content/txnagentbench}"
ARTIFACT_ROOT="${2:-/content/drive/MyDrive/txnagentbench-artifacts}"
OUTPUT_DIR="$ARTIFACT_ROOT/recovery-model-smoke-qwen7b-v1"
CONFIG="configs/recovery_model_smoke_qwen25_7b.json"

cd "$PROJECT_DIR"

if [[ ! -d "$ARTIFACT_ROOT" ]]; then
  echo "Persistent artifact root is missing: $ARTIFACT_ROOT" >&2
  echo "Mount Google Drive before running this script." >&2
  exit 2
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "The Git worktree is dirty. Commit, stash, or discard local edits first." >&2
  exit 2
fi

python - <<'PY'
import torch

assert torch.cuda.is_available(), "A CUDA GPU runtime is required."
properties = torch.cuda.get_device_properties(0)
print("GPU:", torch.cuda.get_device_name(0))
print("GPU memory GiB:", round(properties.total_memory / 2**30, 2))
PY

PROBE="$ARTIFACT_ROOT/.txnagentbench-write-probe"
printf 'persistent-write-ok\n' > "$PROBE"
test "$(cat "$PROBE")" = "persistent-write-ok"
rm "$PROBE"

if [[ -f "$OUTPUT_DIR/artifact_manifest.json" ]]; then
  txnagentbench verify-manifest --output "$OUTPUT_DIR"
  echo "Recovery checkpoint already completed and verified."
  exit 0
fi

python -m pytest -q
txnagentbench plan --config "$CONFIG"
txnagentbench run --config "$CONFIG" --output "$OUTPUT_DIR"
txnagentbench validate --output "$OUTPUT_DIR"

python - "$OUTPUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summaries = [
    json.loads(path.read_text())
    for path in sorted((root / "runs").glob("*/summary.json"))
]

assert len(summaries) == 160, f"Expected 160 runs, found {len(summaries)}"
assert all(row["completed"] for row in summaries)
assert sum(row["invalid_actions"] for row in summaries) == 0
assert sum(row["unsafe_proposals"] for row in summaries) == 0

protected = [
    row for row in summaries
    if row["policy"] in {"txnguard", "mtpo_reproduction"}
]
transaction_only = [
    row for row in summaries
    if row["policy"] in {"serial", "naive", "occ"}
]
assert len(protected) == 64
assert len(transaction_only) == 96
assert all(row["task_success"] for row in protected)
assert all(not row["severe_violation"] for row in protected)
assert all(not row["recovery_violation"] for row in protected)
assert all(row["recovery_rate"] == 1.0 for row in protected)
assert all(row["recovery_violation"] for row in transaction_only)

print("PASS: 160-run real-model recovery checkpoint")
print("Model:", summaries[0]["model_name"])
print("Revision:", summaries[0]["model_revision"])
print("Protected recovery episodes:", sum(row["recovery_attempts"] for row in protected))
print(
    "Transaction-only recovery violations:",
    sum(row["recovery_violation"] for row in transaction_only),
)
PY

bash scripts/finalize_artifact.sh "$OUTPUT_DIR"
echo "PASS: locked real-model recovery artifact: $OUTPUT_DIR"
