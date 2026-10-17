# TxnAgentBench Colab Experiment Runbook

## Stage 0 Research freeze

Before implementation, record the hypotheses, anomaly definitions, primary endpoints, four primary domains, comparison policies, models, exclusion rules, and intended statistical tests. Changes after the pilot must be dated and justified.

## Stage 1 Engineering smoke test

Run the scripted configuration. It should produce 24 completed runs. Confirm that naive parallel execution exposes a lost update under contention while serial, OCC, and TxnGuard remain safe in the minimal scenario.

```bash
txnagentbench run --config configs/smoke.json --output "$ARTIFACT_ROOT/smoke" --allow-scripted
txnagentbench analyze --output "$ARTIFACT_ROOT/smoke"
```

Pass criteria:

- All unit tests pass.
- A second invocation skips all completed runs.
- Every summary has a corresponding trace.
- The same seed produces the same schedule and oracle label.
- A deliberately corrupted trace is rejected by an oracle test.

## Stage 2 Domain implementation

Current status: the in-memory implementations, oracle tests, and deterministic recovery faults
exist. They are suitable for scripted engineering validation. Broader multi-resource scenarios,
adversarial histories, real-model recovery smoke tests, and human oracle agreement remain
publication gates.

Replace the shared-resource demonstration with four independent applications:

1. Inventory reservations with stock and shipment invariants.
2. Calendar scheduling with attendee and room-overlap invariants.
3. Customer-support refunds with exactly-once refund and notification effects.
4. Cloud deployment with dependency ordering and quota invariants.

Each domain needs reset, observe, propose, commit, compensate, and score functions. Add positive and negative histories for every anomaly class.

## Stage 3 Policy implementation

Implement every policy behind the same tool interface. The final study must include serial, naive parallel, PostgreSQL serializable isolation, an independently labelled MTPO reproduction, and TxnGuard. OCC remains a useful secondary baseline. Do not call the MTPO reproduction “CoAgent” or imply that it is author code unless an official immutable artifact becomes available.

Do not rename the included OCC simulation as a PostgreSQL serializable result. The final database baseline must execute real PostgreSQL transactions and record isolation level, transaction retries, SQLSTATE values, and database version.

Before any GPU run, execute `configs/recovery_smoke.json`. It covers pre-commit crashes,
post-commit/pre-ack crashes, delayed acknowledgements, and failed compensation across every domain
and non-database policy. The PostgreSQL CI matrix covers the same faults and must prove both the
successful SQL rollback and the remaining workflow-recovery gap.

## Stage 4 Real model smoke test

Copy the pilot template and set one model ID, immutable model revision, precision, maximum tokens, and prompt version. Run no more than 20 executions.

```bash
txnagentbench run --config configs/pilot_local.json --output "$ARTIFACT_ROOT/model-smoke" --max-runs 20
```

Inspect every failed and invalid action. Do not silently convert malformed model output into a successful action.

After domain or prompt changes, run the locked 16-run cross-domain configuration before the pilot:

```bash
txnagentbench run --config configs/domain_model_smoke_qwen25_7b.json --output "$ARTIFACT_ROOT/domain-model-smoke"
txnagentbench validate --output "$ARTIFACT_ROOT/domain-model-smoke"
txnagentbench analyze --output "$ARTIFACT_ROOT/domain-model-smoke"
```

For the current heterogeneous checkpoint, use the restart-safe wrapper:

```bash
bash scripts/run_diversity_model_smoke.sh /content/txnagentbench "$ARTIFACT_ROOT"
```

This checkpoint covers all four scenario profiles in every domain using the serial policy. It is a
model-decision integration gate, not a policy-effectiveness result.

Then run the 160-cell real-model recovery checkpoint:

```bash
bash scripts/run_recovery_model_smoke.sh /content/txnagentbench "$ARTIFACT_ROOT"
```

This crosses all four domains, five non-database policies, zero/full contention, and all four
workflow-recovery faults. It must have zero invalid and unsafe decisions; TxnGuard and the MTPO
reproduction must complete every triggered recovery without a recovery violation. This is still a
model-smoke artifact, not a primary TMLR result.

## Stage 5 Blinded pilot

Status: completed on 2026-10-16 with 128/128 valid runs. See `docs/PILOT_REPORT.md`. The pilot
passed its engineering gates but exposed insufficient scenario heterogeneity, which must be fixed
before primary freeze.

Run approximately 100 scenario-schedule cells across the candidate policies. Estimate runtime, token use, paired disagreement, severe-violation rate, and infrastructure-failure rate. Use the pilot to calculate the number of final schedule seeds.

Pilot data must be stored separately from the primary experiment and excluded from primary confirmatory estimates.

The post-recovery expanded pilot is split into three prespecified blocks instead of applying every
fault to every domain. This avoids paired no-op cells and reduces the design from 80,640 to 3,264
runs:

- Core: 1,536 runs over ordinary execution and ambiguous-effect replay.
- Cloud dependency: 192 runs where dependency generation is semantically meaningful.
- Recovery: 1,536 runs over the four workflow faults and two actionable scenario families.

All blocks include the real PostgreSQL SSI policy. On Colab, the wrapper installs an ephemeral local
PostgreSQL service, runs the marked integration tests, and records server metadata in every database
summary:

```bash
bash scripts/run_expanded_pilot_qwen7b.sh /content/txnagentbench "$ARTIFACT_ROOT"
```

The earlier 160-run checkpoint used only two agents. After adjusting its measured rate for the
four-agent cells and PostgreSQL retries, budget approximately 60–90 minutes on an A100. Each block
is independently restartable and frozen. These pilot artifacts remain excluded from confirmatory
estimates.

A frozen pilot block may be resumed from a later commit only when the wrapper verifies with Git
that the package source, `pyproject.toml`, and that block's frozen configuration are unchanged.
Each block still records its exact commit. This compatibility exception is limited to pilot
recovery from test/wrapper-only changes; every primary block must use one exact frozen commit.

After the expanded pilot, run the scripted recovery-ablation gate and then the independently pinned
Mistral model checkpoint. The primary configuration must not be frozen until both pass:

```bash
txnagentbench run --config configs/recovery_ablation_smoke.json \
  --output artifacts/recovery-ablation-smoke --allow-scripted

bash scripts/run_mistral_model_smoke.sh \
  /content/txnagentbench "$ARTIFACT_ROOT"
```

Combine frozen pilot blocks only into a separate, mutable analysis directory. Never write a new
analysis file inside a frozen block:

```bash
txnagentbench analyze-study \
  --input "$ARTIFACT_ROOT/expanded-pilot-qwen7b-v2/core" \
  --input "$ARTIFACT_ROOT/expanded-pilot-qwen7b-v2/dependency" \
  --input "$ARTIFACT_ROOT/expanded-pilot-qwen7b-v2/recovery" \
  --output "$ARTIFACT_ROOT/expanded-pilot-qwen7b-v2-analysis"
```

The locked single-model pilot configuration expands to 128 paired engineering runs. Do not execute
it until the 16-run cross-domain model smoke test validates without invalid model actions.

## Stage 6 Primary freeze

Create an immutable primary configuration and record its SHA-256 hash. Tag the code commit. Archive package versions, model revisions, prompts, hardware information, and the analysis script before execution.

Use the built-in preflight and artifact commands rather than notebook-specific hashing code:

```bash
txnagentbench readiness --config configs/primary.json --output "$ARTIFACT_ROOT"
txnagentbench validate --output "$ARTIFACT_ROOT/primary"
txnagentbench freeze --output "$ARTIFACT_ROOT/primary"
txnagentbench verify-manifest --output "$ARTIFACT_ROOT/primary"
```

`readiness` must exit successfully before primary execution. `freeze` is run only after the
artifact is complete and prevents accidental replacement of an existing manifest.

The primary command must refuse to run when:

- A scripted model is configured.
- The documented MTPO reproduction is absent without an approved protocol amendment.
- A model revision is mutable or missing.
- Required oracle tests fail.
- Output storage is not persistent and writable.

## Stage 7 Primary execution

Run in batches of 25 to 50. Randomize the matrix order while preserving paired seeds. Save each run atomically. Synchronize the artifact directory after every batch.

If Colab terminates, reconnect, reinstall the pinned environment, mount the same persistent storage, and rerun the same command. Completed run IDs will be skipped.

## Stage 8 Quality control

Check for missing cells, repeated run IDs, environment changes, cap-triggered failures, invalid model outputs, incomplete traces, and unequal policy exposure. Any exclusion must follow the frozen rules and remain visible in the released run table.

## Stage 9 Analysis

Generate run-level data first, then all tables and figures. Primary results should include paired risk differences, confidence intervals, speedup over serial execution, task-success noninferiority, side-effect violations, token overhead, and recovery rate.

The included analyzer is an engineering check. The final statistical package must add McNemar tests, mixed-effects models, multiplicity handling for secondary comparisons, sensitivity analyses, and human-oracle agreement.

## Stage 10 Release

Release the benchmark code, frozen configurations, scenario seeds, run manifests, permitted traces, result table, analysis scripts, environment lockfile, licenses, and a small reproducibility subset. Remove credentials and identifying metadata from the anonymous TMLR supplement.

