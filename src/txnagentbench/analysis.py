from __future__ import annotations

import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_summaries(output_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((output_dir / "runs").glob("*/summary.json")):
        with path.open("r", encoding="utf-8") as handle:
            rows.append(json.load(handle))
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def result_scope_note(output_dir: Path) -> str:
    config_path = output_dir / "frozen_config.json"
    if not config_path.exists():
        return "Result scope is unknown because the frozen configuration is missing."
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if any(model.get("provider") == "scripted" for model in config.get("models", [])):
        return "Scripted-agent results validate infrastructure only and must not be reported as TMLR findings."
    stage = config.get("stage", "unknown")
    if stage in {"model-smoke", "smoke"}:
        return "Real-model smoke-test results validate model integration only and are not final TMLR findings."
    if stage == "pilot":
        return "Pilot results are for experiment design and must be excluded from primary confirmatory estimates."
    if stage == "primary":
        return "Primary-study results must be interpreted under the frozen protocol and recorded exclusion rules."
    return f"Real-model results from stage '{stage}' are exploratory and are not final TMLR findings."


def bootstrap_interval(values: list[float], samples: int = 2000, seed: int = 1042) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(mean(draw))
    estimates.sort()
    return estimates[int(0.025 * samples)], estimates[min(samples - 1, int(0.975 * samples))]


def finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def exact_mcnemar_pvalue(baseline_only_failure: int, txnguard_only_failure: int) -> float:
    discordant = baseline_only_failure + txnguard_only_failure
    if discordant == 0:
        return 1.0
    tail = min(baseline_only_failure, txnguard_only_failure)
    probability = sum(math.comb(discordant, value) for value in range(tail + 1)) / (2**discordant)
    return min(1.0, 2.0 * probability)


def analyze(output_dir: Path) -> dict[str, Any]:
    rows = load_summaries(output_dir)
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No completed summaries found.")

    run_fields = [
        "run_id",
        "domain",
        "policy",
        "agent_count",
        "contention",
        "scenario_seed",
        "schedule_seed",
        "fault_mode",
        "model_name",
        "model_revision",
        "prompt_version",
        "quantization",
        "accepted",
        "rejected",
        "conflicts",
        "invalid_actions",
        "unsafe_proposals",
        "recovery_attempts",
        "recovery_successes",
        "recovery_rate",
        "recovery_violation",
        "task_success",
        "invariant_violation",
        "side_effect_violation",
        "lost_update",
        "dependency_violation",
        "severe_violation",
        "input_tokens",
        "output_tokens",
        "model_latency_seconds",
        "duration_seconds",
        "side_effect_count",
    ]
    with (analysis_dir / "run_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=run_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in run_fields})

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["policy"]].append(row)
    policy_rows = []
    for policy, items in sorted(grouped.items()):
        recovery_attempts = sum(int(x.get("recovery_attempts", 0)) for x in items)
        recovery_successes = sum(int(x.get("recovery_successes", 0)) for x in items)
        policy_rows.append(
            {
                "policy": policy,
                "runs": len(items),
                "task_success_rate": mean([float(x["task_success"]) for x in items]),
                "severe_violation_rate": mean([float(x["severe_violation"]) for x in items]),
                "lost_update_rate": mean([float(x["lost_update"]) for x in items]),
                "side_effect_violation_rate": mean([float(x["side_effect_violation"]) for x in items]),
                "dependency_violation_rate": mean(
                    [float(x.get("dependency_violation", False)) for x in items]
                ),
                "invalid_action_rate": mean([float(x["invalid_actions"] > 0) for x in items]),
                "unsafe_proposal_rate": mean(
                    [float(x.get("unsafe_proposals", 0) > 0) for x in items]
                ),
                "recovery_attempts": recovery_attempts,
                "recovery_successes": recovery_successes,
                "recovery_rate": (
                    recovery_successes / recovery_attempts if recovery_attempts else None
                ),
                "recovery_violation_rate": mean(
                    [float(x.get("recovery_violation", False)) for x in items]
                ),
                "mean_conflicts": mean([float(x["conflicts"]) for x in items]),
                "median_duration_seconds": statistics.median(x["duration_seconds"] for x in items),
                "mean_total_tokens": mean([x["input_tokens"] + x["output_tokens"] for x in items]),
            }
        )

    with (analysis_dir / "policy_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(policy_rows[0]))
        writer.writeheader()
        writer.writerows(policy_rows)

    domain_policy_rows = []
    by_domain_policy: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_domain_policy[(row["domain"], row["policy"])].append(row)
    for (domain, policy), items in sorted(by_domain_policy.items()):
        domain_policy_rows.append(
            {
                "domain": domain,
                "policy": policy,
                "runs": len(items),
                "task_success_rate": mean([float(x["task_success"]) for x in items]),
                "severe_violation_rate": mean([float(x["severe_violation"]) for x in items]),
                "lost_update_rate": mean([float(x["lost_update"]) for x in items]),
                "side_effect_violation_rate": mean(
                    [float(x["side_effect_violation"]) for x in items]
                ),
            }
        )
    with (analysis_dir / "domain_policy_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(domain_policy_rows[0]))
        writer.writeheader()
        writer.writerows(domain_policy_rows)

    fault_policy_rows = []
    by_fault_policy: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_fault_policy[(row.get("fault_mode", "none"), row["policy"])].append(row)
    for (fault_mode, policy), items in sorted(by_fault_policy.items()):
        recovery_attempts = sum(int(x.get("recovery_attempts", 0)) for x in items)
        recovery_successes = sum(int(x.get("recovery_successes", 0)) for x in items)
        fault_policy_rows.append(
            {
                "fault_mode": fault_mode,
                "policy": policy,
                "runs": len(items),
                "task_success_rate": mean([float(x["task_success"]) for x in items]),
                "severe_violation_rate": mean([float(x["severe_violation"]) for x in items]),
                "lost_update_rate": mean([float(x["lost_update"]) for x in items]),
                "side_effect_violation_rate": mean(
                    [float(x["side_effect_violation"]) for x in items]
                ),
                "recovery_attempts": recovery_attempts,
                "recovery_successes": recovery_successes,
                "recovery_rate": (
                    recovery_successes / recovery_attempts if recovery_attempts else None
                ),
                "recovery_violation_rate": mean(
                    [float(x.get("recovery_violation", False)) for x in items]
                ),
            }
        )
    with (analysis_dir / "fault_policy_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fault_policy_rows[0]))
        writer.writeheader()
        writer.writerows(fault_policy_rows)

    pair_key = lambda row: (
        row["domain"],
        row["agent_count"],
        row["contention"],
        row["model_name"],
        row.get("model_revision"),
        row.get("prompt_version"),
        row.get("quantization"),
        row.get("scenario_seed", 0),
        row["schedule_seed"],
        row.get("fault_mode", "none"),
    )
    by_key: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_key[pair_key(row)][row["policy"]] = row
    paired_diffs = []
    for policies in by_key.values():
        if "txnguard" in policies and "naive" in policies:
            paired_diffs.append(
                float(policies["txnguard"]["severe_violation"])
                - float(policies["naive"]["severe_violation"])
            )
    low, high = bootstrap_interval(paired_diffs)
    paired = {
        "comparison": "txnguard_minus_naive_severe_violation",
        "pairs": len(paired_diffs),
        "risk_difference": finite_or_none(mean(paired_diffs)),
        "ci95_low": finite_or_none(low),
        "ci95_high": finite_or_none(high),
    }
    with (analysis_dir / "paired_effect.json").open("w", encoding="utf-8") as handle:
        json.dump(paired, handle, indent=2, sort_keys=True, allow_nan=False)

    paired_comparisons = []
    for baseline in ["naive", "occ", "serial"]:
        differences = []
        baseline_only_failure = 0
        txnguard_only_failure = 0
        complete_pairs = 0
        for policies in by_key.values():
            if "txnguard" not in policies or baseline not in policies:
                continue
            complete_pairs += 1
            guard_failure = bool(policies["txnguard"]["severe_violation"])
            baseline_failure = bool(policies[baseline]["severe_violation"])
            differences.append(float(guard_failure) - float(baseline_failure))
            baseline_only_failure += int(baseline_failure and not guard_failure)
            txnguard_only_failure += int(guard_failure and not baseline_failure)
        comparison_low, comparison_high = bootstrap_interval(differences)
        paired_comparisons.append(
            {
                "baseline": baseline,
                "pairs": complete_pairs,
                "risk_difference": finite_or_none(mean(differences)),
                "ci95_low": finite_or_none(comparison_low),
                "ci95_high": finite_or_none(comparison_high),
                "baseline_only_failure": baseline_only_failure,
                "txnguard_only_failure": txnguard_only_failure,
                "mcnemar_exact_p": exact_mcnemar_pvalue(
                    baseline_only_failure, txnguard_only_failure
                ),
            }
        )
    with (analysis_dir / "paired_comparisons.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_comparisons[0]))
        writer.writeheader()
        writer.writerows(paired_comparisons)

    report = [
        "# TxnAgentBench analysis",
        "",
        f"Completed runs: {len(rows)}",
        "",
        "## Policy metrics",
        "",
    ]
    for row in policy_rows:
        report.append(
            f"- {row['policy']}: success={row['task_success_rate']:.3f}, "
            f"severe violations={row['severe_violation_rate']:.3f}, runs={row['runs']}"
        )
    report.extend(["", "## Paired engineering comparison", ""])
    if paired_diffs:
        report.append(
            f"TxnGuard minus naive severe-violation risk difference: {paired['risk_difference']:.3f} "
            f"(95% paired bootstrap interval {low:.3f} to {high:.3f}; {paired['pairs']} pairs)."
        )
    else:
        report.append("Not estimable: this artifact contains no paired TxnGuard and naive runs.")
    report.extend(["", result_scope_note(output_dir)])
    (analysis_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return {
        "runs": len(rows),
        "policy_metrics": policy_rows,
        "domain_policy_metrics": domain_policy_rows,
        "fault_policy_metrics": fault_policy_rows,
        "paired": paired,
        "paired_comparisons": paired_comparisons,
    }

