# TxnAgentBench expanded pilot decision report

Status: completed exploratory pilot; excluded from primary confirmatory estimates.

The three immutable blocks contain 3,264/3,264 completed Qwen2.5-7B-Instruct runs at revision
`a09a35458c702b33eeacc393d103063234e8bc28`. The blocks were produced from benchmark commit
`f1993d8ec7d8b0d0057d6654b0348bf4d0144ac5`. No run contained an invalid action or unsafe model
proposal.

## Main observations

- With no injected fault, TxnGuard, OCC, serial execution, PostgreSQL serializable isolation, and
  the MTPO reproduction had zero severe violations. Naive parallel execution had a 0.375 severe
  violation rate across 128 paired cells.
- Under dependency change, TxnGuard and every version-validating or serialized comparator had zero
  severe violations. Naive execution had a 0.75 severe violation rate across 32 paired cells.
- Under duplicate-effect replay, TxnGuard and MTPO had zero severe violations. Every comparator
  without an idempotent effect ledger had a 0.75 severe violation rate.
- Across each of the four workflow-recovery faults, TxnGuard and MTPO recovered every registered
  attempt, while policies intentionally lacking workflow recovery failed every run.
- TxnGuard's median runtime and token use were similar to in-memory OCC and serial execution in this
  implementation. MTPO used more tokens because semantic repair can invoke the model again.

These observations validate the benchmark mechanisms but do not establish a final treatment
effect. In particular, perfect recovery separation is structurally expected when recovery-enabled
systems are compared only with transaction-only systems.

## Mandatory design changes before primary execution

1. Treat concurrency protection and workflow recovery as separate mechanism axes.
2. Add `serial_recovery` and `occ_recovery` controls.
3. Add `recovery_only` and `txnguard_no_recovery` component ablations.
4. Include PostgreSQL and MTPO in every applicable paired-comparison table.
5. Keep no-fault, dependency, replay, and recovery strata separate; do not report a pooled primary
   estimate across mechanistically different faults.
6. Validate a second independently trained model family before freezing primary configurations.
7. Keep all expanded-pilot runs out of primary confidence intervals and hypothesis tests.

The post-pilot code implements these controls and a read-only multi-artifact analyzer. Primary
configuration sizes remain unfrozen until the recovery-ablation smoke test and the second-model
checkpoint both pass.
