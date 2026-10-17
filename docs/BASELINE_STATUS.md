# Publication baseline status

## PostgreSQL Serializable

The database baseline uses real PostgreSQL Serializable Snapshot Isolation through Psycopg 3. Each
agent uses a separate connection and transaction. The gate forces overlapping reads, verifies that
the server reports `serializable`, requires at least one SQLSTATE `40001` retry under contention,
and confirms that the capacity invariant survives. Server version and container image identity must
be recorded with the final artifact.

Run the integration gate with:

```bash
bash scripts/run_postgres_gate.sh
```

The repository CI runs the same tests against a PostgreSQL service on every push. A green unit
suite without the three PostgreSQL tests is insufficient; the integration job must execute them.

The real database engine is connected to the benchmark matrix for `none`, `duplicate_effect`, and
`dependency_change` faults. The gate runs a 96-cell scripted benchmark across all four domains and
scenario families. For the cloud dependency fault, a separate transaction commits a generation
change after all agent transactions read; SSI must abort stale work with `40001`, and the retry must
commit against the current generation.

## CoAgent / MTPO

The CoAgent paper (arXiv:2606.15376) describes Monotonic Trajectory Pre-Order, but the bounded
artifact search on 2026-10-17 found no author-released code, dataset, immutable revision, or
artifact-availability statement. An independently written adapter therefore cannot truthfully be
called the official or faithful CoAgent implementation.

The implemented comparison is named `mtpo_reproduction`. It fixes rank at launch, records physical
trajectory order, notifies agents whose premises changed, invokes the same model for semantic
repair, reconciles operations in rank order, records inverse/reorder counts, and suppresses replayed
effects using registered idempotency keys. Every summary records that it is not official author
code.

Important deviation: the synthetic environments are resettable, so quiescent reconciliation
replays registered operations in rank order rather than performing arbitrary live-system saga
inverses. The paper must report this deviation and use the name `mtpo_reproduction`, not `CoAgent`.
