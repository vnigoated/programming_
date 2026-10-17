#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-/content/txnagentbench}"
ARTIFACT_ROOT="${2:-/content/drive/MyDrive/txnagentbench-artifacts}"
PILOT_ROOT="$ARTIFACT_ROOT/expanded-pilot-qwen7b-v2"

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

python -m pip install -e ".[gpu,analysis,test,postgres]"
bash scripts/setup_colab_postgres.sh
export TXNAGENTBENCH_POSTGRES_DSN="postgresql://txnagentbench:txnagentbench-colab-local@127.0.0.1:5432/txnagentbench"
export TXNAGENTBENCH_TEST_POSTGRES_DSN="$TXNAGENTBENCH_POSTGRES_DSN"

python -m pytest -q

run_block() {
  local name="$1"
  local config="$2"
  local output="$PILOT_ROOT/$name"
  local current_commit
  current_commit="$(git rev-parse HEAD)"

  if [[ -f "$output/artifact_manifest.json" ]]; then
    txnagentbench verify-manifest --output "$output"
    local artifact_commit
    artifact_commit="$(python - "$output/artifact_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
print(manifest["code_commit"])
PY
)"
    if ! git cat-file -e "$artifact_commit^{commit}" 2>/dev/null; then
      echo "Artifact commit is unavailable locally: $artifact_commit" >&2
      exit 2
    fi
    if ! git diff --quiet "$artifact_commit" "$current_commit" -- \
      pyproject.toml src/txnagentbench "$config"; then
      echo "Existing block differs in benchmark code or its frozen configuration." >&2
      echo "Use a new pilot output root instead of mixing implementations." >&2
      exit 2
    fi
    if [[ "$artifact_commit" != "$current_commit" ]]; then
      echo "PASS: benchmark-code compatibility verified: $artifact_commit -> $current_commit"
    fi
    echo "PASS: existing block verified: $name"
    return
  fi

  txnagentbench plan --config "$config"
  txnagentbench run --config "$config" --output "$output"
  txnagentbench validate --output "$output"
  bash scripts/finalize_artifact.sh "$output"
}

run_block "core" "configs/expanded_pilot_core_qwen25_7b.json"
run_block "dependency" "configs/expanded_pilot_dependency_qwen25_7b.json"
run_block "recovery" "configs/expanded_pilot_recovery_qwen25_7b.json"

python - "$PILOT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = {"core": 1536, "dependency": 192, "recovery": 1536}
total_runs = 0
total_invalid = 0
total_unsafe = 0
commits = set()

for name, expected_runs in expected.items():
    block = root / name
    manifest = json.loads((block / "artifact_manifest.json").read_text())
    summaries = [
        json.loads(path.read_text())
        for path in sorted((block / "runs").glob("*/summary.json"))
    ]
    assert len(summaries) == expected_runs, (
        f"{name}: expected {expected_runs} runs, found {len(summaries)}"
    )
    assert manifest["validation"]["completed_runs"] == expected_runs
    commits.add(manifest["code_commit"])
    total_runs += len(summaries)
    total_invalid += sum(row["invalid_actions"] for row in summaries)
    total_unsafe += sum(row["unsafe_proposals"] for row in summaries)
    print(
        f"{name}: runs={len(summaries)}, "
        f"invalid={sum(row['invalid_actions'] for row in summaries)}, "
        f"unsafe={sum(row['unsafe_proposals'] for row in summaries)}"
    )

assert total_runs == 3264
print("PASS: expanded pilot is complete and benchmark-code compatible")
print("Total runs:", total_runs)
print("Invalid decisions retained:", total_invalid)
print("Unsafe proposals retained:", total_unsafe)
print("Recorded block commits:", sorted(commits))
PY

echo "PASS: expanded pilot artifacts: $PILOT_ROOT"
