#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:?usage: finalize_artifact.sh OUTPUT_DIR}"

if [[ -f "$OUTPUT_DIR/artifact_manifest.json" ]]; then
  echo "Artifact is already frozen; verifying it without modification."
  txnagentbench verify-manifest --output "$OUTPUT_DIR"
  exit 0
fi

txnagentbench validate --output "$OUTPUT_DIR"
txnagentbench analyze --output "$OUTPUT_DIR"
txnagentbench validate --output "$OUTPUT_DIR"
txnagentbench freeze --output "$OUTPUT_DIR"
txnagentbench verify-manifest --output "$OUTPUT_DIR"

echo "PASS: artifact validated, analyzed, frozen, and hash-verified"
echo "Manifest: $OUTPUT_DIR/artifact_manifest.json"
