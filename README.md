# TxnAgentBench experiment scaffold

This repository is a restart-safe engineering scaffold for the TxnAgentBench study. The included scripted-agent benchmark validates the scheduler, policy interface, trace format, oracles, checkpointing, and analysis. Scripted-agent results are not research findings.

## What is runnable now

- Four independent in-memory domain simulators for inventory, calendar, support, and cloud
- Domain-specific observations, actions, state transitions, side effects, and correctness oracles
- Independent scenario and schedule seeds
- Paired at-least-once external-effect replay with idempotency tracing
- Serial, naive parallel, optimistic concurrency, and TxnGuard reference policies
- Deterministic schedule seeds
- Immutable per-run JSONL traces
- Atomic per-run summaries and automatic resume
- Policy-level aggregation and paired bootstrap risk difference
- Optional Hugging Face Transformers model adapter
- A primary-study gate that rejects scripted models and stops at the unimplemented CoAgent adapter

The domain simulators validate benchmark semantics, but they are not yet primary-study environments. Before the TMLR experiment, they need broader scenario families, retry/crash injection, adversarial histories, and human oracle validation. PostgreSQL serializable isolation and a faithful CoAgent adapter must also replace the engineering placeholders.

## Local verification

```bash
python -m pip install -e ".[test]"
pytest -q
txnagentbench plan --config configs/smoke.json
txnagentbench run --config configs/smoke.json --output artifacts/smoke --allow-scripted
txnagentbench analyze --output artifacts/smoke
txnagentbench run --config configs/domain_smoke.json --output artifacts/domain-smoke --allow-scripted
txnagentbench validate --output artifacts/domain-smoke
```

Run the full CPU-only diversity gate with `scripts/run_engineering_gates.sh` on Linux/Colab or
`scripts/run_engineering_gates.ps1` on Windows. It covers four scenario families, two agent counts,
three contention levels, two schedules, and normal plus duplicate-effect delivery.

Re-running the same command skips completed run IDs. Delete a single run directory only when intentionally rerunning that exact cell.

## Colab setup

1. Put this repository in a private Git remote.
2. Open `colab/TxnAgentBench_Colab.ipynb` in Colab.
3. Select an A100 runtime.
4. Mount persistent storage.
5. Clone the repository and run `scripts/bootstrap_colab.sh`.
6. Run the scripted smoke test.
7. Run the revision-locked real-model smoke configuration.
8. Inspect traces before increasing the batch size.

The experiment definitions and frozen decisions are in `docs/PROTOCOL.md`. Scripted domain-smoke
results validate the oracles only and must not be reported as empirical findings.

Do not launch the primary matrix until all publication gates in `docs/EXPERIMENT_RUNBOOK.md` pass.

## Artifact layout

```text
artifacts/<stage>/
  environment.json
  frozen_config.json
  progress.json
  runs/<run_id>/
    summary.json
    trace.jsonl
    error.json        # only on failure
  analysis/
    policy_metrics.csv
    domain_policy_metrics.csv
    fault_policy_metrics.csv
    run_table.csv
    paired_comparisons.csv
    paired_effect.json
    report.md
```

## Security

- Store Hugging Face and API tokens in Colab Secrets or environment variables.
- Never commit tokens, private prompts, or user data.
- Use only synthetic benchmark users and mock side effects.
- Do not connect the experiment to real payments, email, cloud, or production databases.

