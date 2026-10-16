from __future__ import annotations

import csv
import json
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


def analyze(output_dir: Path) -> dict[str, Any]:
    rows = load_summaries(output_dir)
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No completed summaries found.")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["policy"]].append(row)
    policy_rows = []
    for policy, items in sorted(grouped.items()):
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
        "risk_difference": mean(paired_diffs),
        "ci95_low": low,
        "ci95_high": high,
    }
    with (analysis_dir / "paired_effect.json").open("w", encoding="utf-8") as handle:
        json.dump(paired, handle, indent=2, sort_keys=True)

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
    report.extend(
        [
            "",
            "## Paired engineering comparison",
            "",
            f"TxnGuard minus naive severe-violation risk difference: {paired['risk_difference']:.3f} "
            f"(95% paired bootstrap interval {low:.3f} to {high:.3f}; {paired['pairs']} pairs).",
            "",
            result_scope_note(output_dir),
        ]
    )
    (analysis_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return {
        "runs": len(rows),
        "policy_metrics": policy_rows,
        "domain_policy_metrics": domain_policy_rows,
        "fault_policy_metrics": fault_policy_rows,
        "paired": paired,
    }

