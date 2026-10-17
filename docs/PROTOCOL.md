# TxnAgentBench frozen protocol draft

Protocol status: engineering freeze for domain implementation. This draft must be versioned and
dated again before the blinded pilot and must receive a final freeze before primary execution.

## Research question

When multiple LLM agents act on shared state, how often do successful-looking local decisions
produce globally invalid state or irreversible duplicate side effects, and how much can a
transaction-aware execution layer reduce those failures without unacceptable loss of task success?

## Unit of analysis and pairing

One run is identified by domain, scenario seed, schedule seed, model revision, agent count,
contention level, injected fault mode, and execution policy. The same cell is paired across policies. Scenario seeds
change task data; schedule seeds change execution order. These sources of variation must not be
conflated. Model and code revisions must be immutable in pilot and primary artifacts.

## Hypotheses

- H1 (primary): TxnGuard has a lower severe-violation probability than naive parallel execution.
- H2 (primary safety/utility): TxnGuard preserves task success within a predeclared
  noninferiority margin relative to serial execution.
- H3 (secondary): TxnGuard reduces duplicate or inconsistent irreversible side effects relative
  to naive execution.
- H4 (secondary): TxnGuard improves throughput or latency relative to serial execution at matched
  task success and safety.

The noninferiority margin and target sample size are chosen after the blinded pilot and frozen
before primary execution. Pilot runs are excluded from confirmatory estimates.

## Primary endpoint

`severe_violation` is true when any domain oracle detects an invariant violation, an externally
visible side-effect violation, a lost update, a dependency-order violation, or an unrecovered
workflow fault. The primary effect is the paired risk difference:

`P(severe | TxnGuard) - P(severe | naive)`.

Negative values favor TxnGuard. All run-level outcomes remain in the released table, including
model-format failures and infrastructure failures.

## Domains and independent oracles

### Inventory reservation

Agents reserve scarce stock. State contains stock, reservation count, and a version. The external
effect is shipment creation. Oracles detect reservations above stock, shipments above stock, and
shipment/ledger disagreement.

### Calendar scheduling

Agents book exclusive room-time slots. State contains concrete meeting identifiers rather than a
numeric allocation ledger. The external effect is invitation delivery. Oracles detect overlapping
bookings, multiple invitation sets for one exclusive slot, and invitation/calendar disagreement.

### Customer-support refunds

Agents issue refunds against a case-specific eligible amount. State is recorded in cents and the
external effects are a payment refund and customer notification. Oracles detect ledger amounts
above eligibility, refunded money above eligibility, unmatched notifications, and payment/ledger
disagreement.

### Cloud deployment

Agents deploy services against quota while referencing a dependency generation. State contains
quota usage, dependency readiness, generation, and a version. The external effect is deployment
creation. Oracles detect quota overflow, deployments above quota, ledger/deployment disagreement,
and deployment against the wrong dependency generation.

## Fault modes

- `none`: no injected delivery failure.
- `duplicate_effect`: after a successful commit, the external-effect delivery is replayed with the
  same idempotency key, modeling at-least-once delivery after an ambiguous acknowledgement. Serial
  execution and OCC do not inherently deduplicate this external replay. TxnGuard's effect ledger
  must suppress it. The trace records both the replay attempt and whether delivery occurred.
- `dependency_change`: after parallel agents observe cloud state but before they commit, the cloud
  dependency generation advances and the resource version changes. Naive execution may deploy
  against the obsolete generation; version-validating policies must refresh or reject. Other
  domains treat this fault as a paired no-op control.
- `crash_before_commit`: the worker stops after deciding but before a commit. The TxnGuard reference
  policy resumes from its registered recovery record, and the MTPO reproduction resumes from its
  registered trajectory. The PostgreSQL baseline performs its
  SQL update and then raises inside the transaction, proving atomic rollback while also recording
  that database atomicity alone does not recover the lost workflow job.
- `crash_after_commit_before_ack`: the commit succeeds, the worker stops before acknowledging it,
  and the operation is retried. Policies without an external idempotency ledger redeliver the
  effect.
- `delayed_ack`: the acknowledgement times out and the same operation is retried. This shares the
  replay mechanism with the post-commit crash but records a distinct causal event.
- `compensation_failure`: the first registered inverse operation fails. TxnGuard and the MTPO
  reproduction retry the inverse; transaction-only policies leave the committed effect in place.

Fault injection is deterministic and paired across policies; it must never depend on an observed
treatment result.

## Policies

- `serial`: each agent observes immediately before an atomic commit.
- `naive`: agents decide from pre-execution snapshots and write without conflict validation.
- `occ`: stale versions conflict; the system refreshes and rejects an infeasible action.
- `txnguard`: version validation, domain invariant checks, registered-resume semantics, idempotent
  side-effect keys, and retryable compensation. The in-memory recovery record is a benchmark
  abstraction, not evidence of process-level durability.
- `postgres_serializable`: real PostgreSQL SSI transactions with SQLSTATE `40001` retries. It
  guarantees database atomicity but intentionally has no workflow/effect recovery layer.
- `mtpo_reproduction`: independently implemented rank-ordered trajectory reconciliation with
  semantic repair, guarded replay, and registered inverse retry. It is not author-released CoAgent
  code and must not be described as such.

The in-memory OCC policy must never be described as PostgreSQL serializable isolation.

## Model behavior

The model receives only its observed snapshot and task request. It may apply or reject. An applied
amount must exactly equal the requested amount; otherwise it is an invalid action. Parsing failures
remain failures. The model prompt, decoding parameters, identifier, and immutable model revision
are frozen with each experiment.

Prompt version `allocation-v2` supplies a tool-computed `available` value and requires rejection
when a positive request exceeds it. A syntactically valid `apply` that exceeds observed availability
is recorded separately as an `unsafe_proposal`; it is not silently relabeled as a parser failure.

## Metrics

Primary metrics are severe-violation rate and task-success rate. Secondary metrics include lost
updates, side-effect violations, dependency violations, conflict/retry counts, invalid actions,
recovery attempts, recovery successes, recovery-violation rate, latency, and token use. Recovery
rate is the aggregate successful recoveries divided by recovery attempts. Throughput measurements
exclude model loading but include model inference and policy execution.

## Statistical plan

The primary comparison uses paired cells. Report paired risk differences with confidence
intervals and a paired binary test. The final analysis will use a mixed-effects model with domain,
model family, agent count, and contention as prespecified factors, with scenario as a random
effect. Secondary comparisons require multiplicity control. Report sensitivity analyses that
count infrastructure failures as failures and that exclude them under the frozen exclusion rule.

## Exclusions and stopping

Exclude a run only for a documented infrastructure failure that occurs before an agent decision;
retain malformed outputs, timeouts after dispatch, and safety failures. Do not inspect treatment
effects while the primary run is incomplete. Stop for corrupted artifacts, non-immutable model
revision, changed code/config hashes, or an oracle validation failure—not for an unfavorable
result.

## Current engineering limitations

The deterministic scripted-agent matrices support oracle and recovery-infrastructure validation,
not paper claims. Before the primary study, add broader multi-resource scenarios, adversarial
histories, human oracle agreement, real-model recovery smoke tests, and the final statistical
analysis package.
