# Heterogeneous real-model checkpoint report

Date: 2026-10-17  
Status: passed; model-integration evidence only, excluded from primary confirmatory estimates

## Frozen inputs

- Code commit: `72e2aa5f42305af56cb8cff061123546847db289`
- Model: `Qwen/Qwen2.5-7B-Instruct`
- Model revision: `a09a35458c702b33eeacc393d103063234e8bc28`
- Precision: BF16
- Prompt: `allocation-v2`
- Configuration: `configs/diversity_model_smoke_qwen25_7b.json`
- Domains: inventory, calendar, support, and cloud
- Policy: serial
- Scenario profiles per domain: scarce, two-fit, partially allocated, and exhausted

## Execution and integrity

The checkpoint completed 16 of 16 planned real-model runs. Artifact validation checked all 16
summary/trace pairs and reported no missing, unexpected, or inconsistent runs. The model produced
zero invalid actions and zero unsafe proposals. All runs achieved task success with no invariant,
lost-update, side-effect, dependency, or severe violation.

The final artifact contains 42 hashed files and passed manifest verification. It is stored at
`txnagentbench-artifacts/diversity-model-smoke-qwen7b` in persistent experiment storage. The
manifest records the code and model revisions above.

## Aggregate integration metrics

| Policy | Runs | Task success | Severe violations | Invalid actions | Unsafe proposals | Median duration | Mean tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| serial | 16 | 1.00 | 0.00 | 0.00 | 0.00 | 0.296 s | 525.75 |

Each of the four domains completed four runs with full task success and no severe violations.

## Interpretation

This result verifies that the locked model and prompt correctly handle every current domain schema
and all four heterogeneous capacity profiles under sequential execution. It does not compare
concurrency policies and therefore provides no estimate of TxnGuard effectiveness. The absent
paired-policy estimate is now represented as JSON `null`, rather than non-standard `NaN`, in newer
analysis outputs.
