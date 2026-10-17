# TxnAgentBench experiment scaffold

This repository is a restart-safe engineering scaffold for the TxnAgentBench study. The included scripted-agent benchmark validates the scheduler, policy interface, trace format, oracles, checkpointing, and analysis. Scripted-agent results are not research findings.

## What is runnable now

- Four independent in-memory domain simulators for inventory, calendar, support, and cloud
- Domain-specific observations, actions, state transitions, side effects, and correctness oracles
- Independent scenario and schedule seeds
- Seven paired fault modes covering concurrency, dependency changes, ambiguous acknowledgements,
  worker crashes, replayed effects, and compensation failure
- Serial, naive parallel, optimistic concurrency, and TxnGuard reference policies
- Deterministic schedule seeds
- Immutable per-run JSONL traces
- Atomic per-run summaries and automatic resume
- Built-in artifact validation, SHA-256 freezing, and later integrity verification
- Explicit primary-study readiness report that fails closed on missing baselines
- Policy-level aggregation and paired bootstrap risk difference
- Optional Hugging Face Transformers model adapter
- Real PostgreSQL SSI and independently labelled MTPO reproduction baselines
- A primary-study gate that rejects scripted models, mutable revisions, omitted baselines, and
  omitted fault modes

The scripted matrices validate benchmark semantics and recovery infrastructure; they are not TMLR
findings. Before the primary experiment, the study still needs broader multi-resource scenarios,
adversarial histories, human oracle validation, real-model recovery smoke tests, and final
statistical code. PostgreSQL serializable isolation and an independently labelled MTPO reproduction
are implemented. No official CoAgent artifact was available in the bounded search, so the project
does not claim its reproduction is the authors' implementation.

## Local verification

```bash
python -m pip install -e ".[test]"
pytest -q
txnagentbench plan --config configs/smoke.json
txnagentbench run --config configs/smoke.json --output artifacts/smoke --allow-scripted
txnagentbench analyze --output artifacts/smoke
txnagentbench run --config configs/domain_smoke.json --output artifacts/domain-smoke --allow-scripted
txnagentbench validate --output artifacts/domain-smoke
txnagentbench run --config configs/recovery_smoke.json --output artifacts/recovery-smoke --allow-scripted
txnagentbench validate --output artifacts/recovery-smoke
txnagentbench analyze --output artifacts/recovery-smoke
txnagentbench freeze --output artifacts/domain-smoke
txnagentbench verify-manifest --output artifacts/domain-smoke
```

Run the full CPU-only diversity gate with `scripts/run_engineering_gates.sh` on Linux/Colab or
`scripts/run_engineering_gates.ps1` on Windows. It covers four scenario families, two agent counts,
three contention levels, two schedules, and a separate 320-cell workflow-recovery matrix.

After reconnecting a GPU Colab runtime, the next locked checkpoint is one command:

```bash
bash scripts/run_diversity_model_smoke.sh \
  /content/txnagentbench \
  /content/drive/MyDrive/txnagentbench-artifacts
```

It probes persistent storage, runs all tests, executes the locked 16-run heterogeneous Qwen
checkpoint, validates and analyzes it, freezes every file with SHA-256, verifies the manifest,
and enforces zero invalid/unsafe decisions. A frozen output directory cannot be resumed or altered.

After that heterogeneous decision checkpoint passes, run the locked real-model workflow-recovery
checkpoint:

```bash
bash scripts/run_recovery_model_smoke.sh \
  /content/txnagentbench \
  /content/drive/MyDrive/txnagentbench-artifacts
```

It executes 160 runs across all four domains, five non-database policies, zero and full contention,
and the four workflow faults. It performs quality checks before freezing the artifact, so a failed
gate remains inspectable and restartable.

Once that checkpoint passes, the next prespecified pilot is:

```bash
bash scripts/run_expanded_pilot_qwen7b.sh \
  /content/txnagentbench \
  /content/drive/MyDrive/txnagentbench-artifacts
```

The wrapper starts a local PostgreSQL service and runs three independently frozen blocks totaling
3,264 real-model runs. Pilot outputs are excluded from final confirmatory estimates.

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
Run `txnagentbench readiness --config configs/primary_template.json --output /mounted/artifacts` to see the unresolved gates;
its nonzero exit status is intentional until the real PostgreSQL and MTPO reproduction adapters exist.

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

