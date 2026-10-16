# TxnAgentBench blinded pilot report

Date: 2026-10-16  
Status: exploratory engineering pilot; excluded from primary confirmatory estimates

## Frozen inputs

- Code commit: `bcd023aa64ea589e666aaed1c473419fccb90df4`
- Model: `Qwen/Qwen2.5-7B-Instruct`
- Model revision: `a09a35458c702b33eeacc393d103063234e8bc28`
- Precision: BF16
- Prompt: `allocation-v2`
- Configuration: `configs/pilot_qwen25_7b.json`
- Domains: inventory, calendar, support, and cloud
- Policies: serial, naive, OCC, and TxnGuard
- Fault modes: none and duplicate external-effect delivery

## Integrity and execution

The pilot planned and completed 128 of 128 runs with zero runner failures. Artifact validation
checked all 128 summary/trace pairs and found no missing, unexpected, or inconsistent runs. The
model produced zero invalid actions and zero unsafe proposals.

## Aggregate exploratory results

| Policy | Runs | Task success | Severe violations | Lost updates | Side-effect violations |
|---|---:|---:|---:|---:|---:|
| naive | 32 | 0.25 | 0.75 | 0.75 | 0.75 |
| OCC | 32 | 0.50 | 0.50 | 0.50 | 0.50 |
| serial | 32 | 0.50 | 0.50 | 0.50 | 0.50 |
| TxnGuard | 32 | 1.00 | 0.00 | 0.00 | 0.00 |

The exploratory paired TxnGuard-minus-naive severe-violation risk difference was -0.75, with an
engineering bootstrap interval of [-0.906, -0.594] across 32 paired cells. This interval is not a
confirmatory confidence interval because the scenario cells are deliberately small, highly
structured, and not an empirical sample from a defined deployment population.

## Fault-stratified interpretation

Without injected effect replay, serial, OCC, and TxnGuard had full task success and no severe
violations. Naive execution failed in the contended half of its cells. Under duplicate-effect
delivery, serial, OCC, and naive execution failed every cell because none supplied an external
idempotency ledger. TxnGuard suppressed every replay and remained safe.

Median run duration was approximately 0.62 seconds on the pilot hardware. Mean prompt plus output
tokens were approximately 503 per run. Model loading was excluded from run duration.

## What the pilot validates

- Revision-locked real-model inference across all four schemas
- Prompt adherence for both available and exhausted resources
- Paired policy execution and automatic resume
- Domain-specific state and side-effect oracles
- At-least-once effect replay and TxnGuard idempotency behavior
- Artifact integrity validation and fault-stratified analysis

## Why the primary study is not ready

All four domains produced identical aggregate rates. The current scenario seeds change concrete
values, but not enough causal structure or difficulty to create meaningful heterogeneity. A larger
run over the same generators would create narrower-looking intervals without adding independent
evidence. Dependency violations also remained at zero because dependency-change faults are not yet
injected.

The primary study remains blocked on:

1. Broader scenario families with heterogeneous capacities, pre-existing allocations, request
   sizes, valid rejections, and multi-resource constraints.
2. Crash points before and after state commit, dependency changes, delayed acknowledgements, and
   compensation failures.
3. Agent counts above two and intermediate contention.
4. A real PostgreSQL serializable baseline with retry and SQLSTATE logging.
5. A faithful reproducible multi-agent baseline or a dated protocol amendment.
6. At least one additional immutable model family.
7. Human review of a stratified oracle sample and a frozen primary statistical analysis.

## Decision

The engineering pilot passes. Its data will be retained and released as pilot evidence, but excluded
from primary hypothesis tests. No primary execution should begin until the gates above pass.
