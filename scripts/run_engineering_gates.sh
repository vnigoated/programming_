#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-$(pwd)}"
OUTPUT_DIR="${2:-$PROJECT_DIR/artifacts/diversity-smoke}"
RECOVERY_OUTPUT_DIR="${3:-$PROJECT_DIR/artifacts/recovery-smoke}"

cd "$PROJECT_DIR"
python -m pip install -e ".[test]"
pytest -q
txnagentbench plan --config configs/diversity_smoke.json
txnagentbench run --config configs/diversity_smoke.json --output "$OUTPUT_DIR" --allow-scripted
txnagentbench validate --output "$OUTPUT_DIR"
txnagentbench analyze --output "$OUTPUT_DIR"
txnagentbench plan --config configs/recovery_smoke.json
txnagentbench run --config configs/recovery_smoke.json --output "$RECOVERY_OUTPUT_DIR" --allow-scripted
txnagentbench validate --output "$RECOVERY_OUTPUT_DIR"
txnagentbench analyze --output "$RECOVERY_OUTPUT_DIR"
echo "Engineering gates passed: $OUTPUT_DIR and $RECOVERY_OUTPUT_DIR"
