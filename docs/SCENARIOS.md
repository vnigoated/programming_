# Scenario catalog

Scenario seeds are structural benchmark inputs, not repeated random-number labels. The current
catalog maps `scenario_seed % 4` to one of four prespecified families.

| Seed family | Capacity | Initial allocation | Request | Intended behavior |
|---|---:|---:|---:|---|
| 0: scarce | 1 | 0 | 1 | Only one contending request can commit |
| 1: two-fit | 2 | 0 | 1 | Two requests may commit, exposing stale overwrite rather than oversubscription |
| 2: partially allocated | 3 | 1 | 2 | One large request fits after a baseline allocation |
| 3: exhausted | 1 | 1 | 1 | Correct behavior is rejection by every new agent |

Inventory uses units directly. Support scales each unit to 2,500 cents. Cloud uses quota units and
marks the exhausted family dependency unavailable. Calendar remains exclusive-capacity but varies
meeting duration; family 3 begins with a baseline booking and invitation.

Baseline state has matching baseline side effects and is excluded from measured side-effect count.
Oracles still include it when checking ledger/effect consistency. Historical baseline cloud
deployments are not judged against a later dependency generation.

## Contention

At zero contention every agent receives a private resource. At positive contention, a deterministic
fraction of agents share one resource: at least two agents, capped by agent count. The remaining
agents receive private resources. Runs cover 0%, 50%, and 100% requested contention with two and
four agents.

## Faults

- `none`: ordinary execution.
- `duplicate_effect`: replay the same external idempotency key after commit.
- `dependency_change`: advance cloud dependency generation after parallel reads and before commits;
  it is a paired no-op control in other domains.

## Remaining scenario work

Before primary freeze, add multi-resource write-skew cases, crash-before/after-commit recovery,
delayed acknowledgements, compensation failure, and human-reviewed natural-language task variants.
