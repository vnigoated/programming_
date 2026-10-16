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

Current status: the first in-memory implementations and oracle tests exist. They are suitable for
scripted engineering validation. Broader scenario families, injected retries/crashes, adversarial
histories, and human oracle agreement remain publication gates.

Replace the shared-resource demonstration with four independent applications:

1. Inventory reservations with stock and shipment invariants.
2. Calendar scheduling with attendee and room-overlap invariants.
3. Customer-support refunds with exactly-once refund and notification effects.
4. Cloud deployment with dependency ordering and quota invariants.

Each domain needs reset, observe, propose, commit, compensate, and score functions. Add positive and negative histories for every anomaly class.

## Stage 3 Policy implementation

Implement every policy behind the same tool interface. The final study must include serial, naive parallel, PostgreSQL serializable isolation, a faithful CoAgent adapter when reproducible, and TxnGuard. OCC remains a useful secondary baseline.

Do not rename the included OCC simulation as a PostgreSQL serializable result. The final database baseline must execute real PostgreSQL transactions and record isolation level, transaction retries, SQLSTATE values, and database version.

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

## Stage 5 Blinded pilot

Status: completed on 2026-10-16 with 128/128 valid runs. See `docs/PILOT_REPORT.md`. The pilot
passed its engineering gates but exposed insufficient scenario heterogeneity, which must be fixed
before primary freeze.

Run approximately 100 scenario-schedule cells across the candidate policies. Estimate runtime, token use, paired disagreement, severe-violation rate, and infrastructure-failure rate. Use the pilot to calculate the number of final schedule seeds.

Pilot data must be stored separately from the primary experiment and excluded from primary confirmatory estimates.

The locked single-model pilot configuration expands to 128 paired engineering runs. Do not execute
it until the 16-run cross-domain model smoke test validates without invalid model actions.

## Stage 6 Primary freeze

Create an immutable primary configuration and record its SHA-256 hash. Tag the code commit. Archive package versions, model revisions, prompts, hardware information, and the analysis script before execution.

The primary command must refuse to run when:

- A scripted model is configured.
- CoAgent is absent without an approved protocol amendment.
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

