# TxnAgentBench frozen protocol draft

Protocol status: engineering freeze for domain implementation. This draft must be versioned and
dated again before the blinded pilot and must receive a final freeze before primary execution.

## Research question

When multiple LLM agents act on shared state, how often do successful-looking local decisions
produce globally invalid state or irreversible duplicate side effects, and how much can a
transaction-aware execution layer reduce those failures without unacceptable loss of task success?

## Unit of analysis and pairing

One run is identified by domain, scenario seed, schedule seed, model revision, agent count,
contention level, and execution policy. The same cell is paired across policies. Scenario seeds
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
visible side-effect violation, a lost update, or a dependency-order violation. The primary effect
is the paired risk difference:

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

## Policies

- `serial`: each agent observes immediately before an atomic commit.
- `naive`: agents decide from pre-execution snapshots and write without conflict validation.
- `occ`: stale versions conflict; the system refreshes and rejects an infeasible action.
- `txnguard`: version validation plus domain invariant checks and idempotent side-effect keys.
- PostgreSQL serializable: publication baseline still to be implemented against a real database.
- CoAgent: publication baseline still to be implemented faithfully or removed only through a
  dated protocol amendment justified by reproducibility evidence.

The in-memory OCC policy must never be described as PostgreSQL serializable isolation.

## Model behavior

The model receives only its observed snapshot and task request. It may apply or reject. An applied
amount must exactly equal the requested amount; otherwise it is an invalid action. Parsing failures
remain failures. The model prompt, decoding parameters, identifier, and immutable model revision
are frozen with each experiment.

## Metrics

Primary metrics are severe-violation rate and task-success rate. Secondary metrics include lost
updates, side-effect violations, dependency violations, conflict/retry counts, invalid actions,
latency, token use, and recovery rate. Throughput measurements exclude model loading but include
model inference and policy execution.

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

The four in-memory simulators and deterministic scripted agent support oracle validation, not
paper claims. Before the primary study, add broader scenario families, adversarial histories,
human oracle agreement, real PostgreSQL serializable transactions, the selected multi-agent
baseline, crash/retry injection, and final statistical analysis.
