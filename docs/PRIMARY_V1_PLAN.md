# TxnAgentBench primary-v1 frozen design

The primary design is split into three immutable blocks and contains 15,040 runs:

| Block | Runs | Purpose |
| --- | ---: | --- |
| concurrency | 6,912 | Confirmatory TxnGuard-versus-naive severe-violation comparison without injected faults |
| dependency | 1,728 | Cloud dependency-generation changes |
| recovery | 6,400 | Replay, crash, delayed acknowledgement, and compensation mechanisms plus ablations |

Both Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.3 use immutable model revisions. The same
scenario and schedule cells are paired across policies. The recovery block includes transaction-only
comparators, component-matched recovery controls, `recovery_only`, and
`txnguard_no_recovery`.

## Confirmatory and secondary inference

The sole confirmatory hypothesis is the paired severe-violation risk difference between TxnGuard
and naive parallel execution in the no-fault concurrency block. The dependency and recovery
comparisons are prespecified secondary hypotheses and use Holm adjustment. Task-success
noninferiority uses a five-percentage-point margin against serial execution in the concurrency
block and against `serial_recovery` in the recovery block.

Paired bootstrap confidence intervals use 10,000 resamples with seed 1701. Exact McNemar tests use
the paired discordant cells. A binomial mixed-effects sensitivity analysis includes fixed effects
for policy, domain, model, agent count, contention, and fault, with a scenario-level random
intercept. Failure to estimate the mixed model under complete separation is reported rather than
silently changing the model.

## Blinding and integrity

Pilot artifacts are excluded. Primary blocks must all execute at the Git tag
`txnagentbench-primary-v1-freeze`; no compatibility exception is permitted. Each block is frozen
and hash-verified separately. Analysis writes to a new directory outside the immutable artifacts.
The human-oracle audit follows `configs/primary_v1_oracle_audit_plan.json` and is performed without
showing annotators policy names or automated oracle outcomes.

No primary run may begin until the suite validator, environmental readiness gate, complete test
suite, PostgreSQL integration tests, and both model checkpoints pass.

## Frozen file hashes

| File | SHA-256 |
| --- | --- |
| `primary_v1_concurrency.json` | `4a0cff2d0c78b7d74b20c6c714fe87f9d7b864c959ab871b4c602c273cd94c49` |
| `primary_v1_dependency.json` | `e7f752fd81741d4f9b13be5d6766f68d86745eb1fbe2fdbd98792114632947d9` |
| `primary_v1_recovery.json` | `399a052becc306299b74b5c57d2bf2edb44990f2d47e5f7997e2c5a25f2138df` |
| `primary_v1_analysis_plan.json` | `5bb799b669d98790d3e50c531add12f5957a914f3381488426e723c7d9ba1adf` |
| `primary_v1_oracle_audit_plan.json` | `c97d280d6abc424089efb7d91d8b9590e43bc3c7b6c7c94b48a3265a38ecd12b` |
