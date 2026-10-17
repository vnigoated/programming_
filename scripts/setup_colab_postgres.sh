#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

if ! command -v psql >/dev/null 2>&1; then
  apt-get -qq update
  apt-get -qq install -y postgresql postgresql-contrib
fi

service postgresql start

if [[ "$(runuser -u postgres -- psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='txnagentbench'")" != "1" ]]; then
  runuser -u postgres -- psql --set ON_ERROR_STOP=1 --command \
    "CREATE ROLE txnagentbench LOGIN PASSWORD 'txnagentbench-colab-local';"
fi

if [[ "$(runuser -u postgres -- psql -tAc "SELECT 1 FROM pg_database WHERE datname='txnagentbench'")" != "1" ]]; then
  runuser -u postgres -- createdb --owner=txnagentbench txnagentbench
fi

pg_isready -h 127.0.0.1 -p 5432 -d txnagentbench -U txnagentbench
echo "PASS: local PostgreSQL is ready for TxnAgentBench"
