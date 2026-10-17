#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-$(pwd)}"
COMPOSE_FILE="$PROJECT_DIR/docker-compose.postgres.yml"
export TXNAGENTBENCH_TEST_POSTGRES_DSN="postgresql://txnagentbench:txnagentbench-local-only@127.0.0.1:55432/txnagentbench"

cd "$PROJECT_DIR"

cleanup() {
  docker compose -f "$COMPOSE_FILE" down --volumes
}
trap cleanup EXIT

docker compose -f "$COMPOSE_FILE" up -d --wait
python -m pip install -e ".[postgres,test]"
python -m pytest -q -m postgres
txnagentbench plan --config configs/postgres_smoke.json
TXNAGENTBENCH_POSTGRES_DSN="$TXNAGENTBENCH_TEST_POSTGRES_DSN" \
  txnagentbench run \
  --config configs/postgres_smoke.json \
  --output artifacts/postgres-smoke-gate \
  --allow-scripted
txnagentbench validate --output artifacts/postgres-smoke-gate
txnagentbench analyze --output artifacts/postgres-smoke-gate

docker compose -f "$COMPOSE_FILE" images
echo "PASS: real PostgreSQL serializable integration and benchmark gates"
