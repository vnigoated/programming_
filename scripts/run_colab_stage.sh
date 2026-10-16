#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:?usage: run_colab_stage.sh CONFIG OUTPUT_DIR [MAX_RUNS]}"
OUTPUT_DIR="${2:?usage: run_colab_stage.sh CONFIG OUTPUT_DIR [MAX_RUNS]}"
MAX_RUNS="${3:-}"

ARGS=(run --config "$CONFIG" --output "$OUTPUT_DIR")
if [[ -n "$MAX_RUNS" ]]; then
  ARGS+=(--max-runs "$MAX_RUNS")
fi
if grep -q '"provider"[[:space:]]*:[[:space:]]*"scripted"' "$CONFIG"; then
  ARGS+=(--allow-scripted)
fi

txnagentbench "${ARGS[@]}"
txnagentbench analyze --output "$OUTPUT_DIR"

