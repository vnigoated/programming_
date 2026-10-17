#!/usr/bin/env bash
set -euo pipefail

BLOCK="${1:?usage: run_primary_v1_batch.sh BLOCK [PROJECT_DIR] [ARTIFACT_ROOT] [BATCH_SIZE]}"
PROJECT_DIR="${2:-/content/txnagentbench}"
ARTIFACT_ROOT="${3:-/content/drive/MyDrive/txnagentbench-artifacts}"
BATCH_SIZE="${4:-250}"
PRIMARY_ROOT="$ARTIFACT_ROOT/txnagentbench-primary-v1"

case "$BLOCK" in
  concurrency)
    CONFIG="configs/primary_v1_concurrency.json"
    EXPECTED_RUNS=6912
    ;;
  dependency)
    CONFIG="configs/primary_v1_dependency.json"
    EXPECTED_RUNS=1728
    ;;
  recovery)
    CONFIG="configs/primary_v1_recovery.json"
    EXPECTED_RUNS=6400
    ;;
  *)
    echo "Unknown primary block: $BLOCK" >&2
    exit 2
    ;;
esac

if ! [[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "BATCH_SIZE must be a positive integer." >&2
  exit 2
fi

OUTPUT_DIR="$PRIMARY_ROOT/$BLOCK"
cd "$PROJECT_DIR"

if [[ ! -d "$ARTIFACT_ROOT" ]]; then
  echo "Persistent artifact root is missing: $ARTIFACT_ROOT" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "The Git worktree is dirty. Primary experiments require the exact frozen commit." >&2
  exit 2
fi
if ! git rev-parse --verify --quiet "refs/tags/txnagentbench-primary-v1-freeze" >/dev/null; then
  echo "Missing required primary freeze tag: txnagentbench-primary-v1-freeze" >&2
  exit 2
fi
EXPECTED_COMMIT="$(git rev-list -n 1 txnagentbench-primary-v1-freeze)"
CURRENT_COMMIT="$(git rev-parse HEAD)"
if [[ "$CURRENT_COMMIT" != "$EXPECTED_COMMIT" ]]; then
  echo "Primary batches must run at tag txnagentbench-primary-v1-freeze." >&2
  echo "Expected: $EXPECTED_COMMIT" >&2
  echo "Current:  $CURRENT_COMMIT" >&2
  exit 2
fi
echo "Primary freeze commit: $CURRENT_COMMIT"

python - <<'PY'
import torch

assert torch.cuda.is_available(), "A CUDA GPU runtime is required."
print("GPU:", torch.cuda.get_device_name(0))
print("GPU memory GiB:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
PY

python -m pip install -e ".[gpu,analysis,test,postgres]"
bash scripts/setup_colab_postgres.sh
export TXNAGENTBENCH_POSTGRES_DSN="postgresql://txnagentbench:txnagentbench-colab-local@127.0.0.1:5432/txnagentbench"
export TXNAGENTBENCH_TEST_POSTGRES_DSN="$TXNAGENTBENCH_POSTGRES_DSN"

python -m pytest -q
txnagentbench validate-primary-suite \
  --config configs/primary_v1_concurrency.json \
  --config configs/primary_v1_dependency.json \
  --config configs/primary_v1_recovery.json
txnagentbench readiness --config "$CONFIG" --output "$PRIMARY_ROOT"

if [[ -f "$OUTPUT_DIR/artifact_manifest.json" ]]; then
  txnagentbench verify-manifest --output "$OUTPUT_DIR"
  echo "PASS: primary block already complete and verified: $BLOCK"
  exit 0
fi

txnagentbench run \
  --config "$CONFIG" \
  --output "$OUTPUT_DIR" \
  --max-runs "$BATCH_SIZE"

COMPLETED="$(python - "$OUTPUT_DIR" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
print(sum(1 for _ in (root / "runs").glob("*/summary.json")))
PY
)"
ERRORS="$(python - "$OUTPUT_DIR" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
print(sum(1 for _ in (root / "runs").glob("*/error.json")))
PY
)"

echo "Primary block progress: $BLOCK $COMPLETED/$EXPECTED_RUNS; pending errors=$ERRORS"

if [[ "$COMPLETED" -eq "$EXPECTED_RUNS" && "$ERRORS" -eq 0 ]]; then
  txnagentbench validate --output "$OUTPUT_DIR"
  bash scripts/finalize_artifact.sh "$OUTPUT_DIR"
  echo "PASS: primary block complete, frozen, and verified: $BLOCK"
else
  echo "Batch checkpoint saved. Rerun the same command to continue."
fi
