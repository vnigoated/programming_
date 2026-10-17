from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .analysis import (
    bootstrap_interval,
    comparison_baselines,
    exact_mcnemar_pvalue,
    finite_or_none,
    load_summaries,
    mean,
)


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _pair_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["artifact"],
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


def _paired_row(
    items: list[dict[str, Any]], baseline: str, strata: dict[str, Any]
) -> dict[str, Any]:
    by_key: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in items:
        by_key[_pair_key(row)][row["policy"]] = row
    differences: list[float] = []
    baseline_only_failure = 0
    txnguard_only_failure = 0
    for policies in by_key.values():
        if baseline not in policies or "txnguard" not in policies:
            continue
        baseline_failure = bool(policies[baseline]["severe_violation"])
        txnguard_failure = bool(policies["txnguard"]["severe_violation"])
        differences.append(float(txnguard_failure) - float(baseline_failure))
        baseline_only_failure += int(baseline_failure and not txnguard_failure)
        txnguard_only_failure += int(txnguard_failure and not baseline_failure)
    low, high = bootstrap_interval(differences)
    return {
        **strata,
        "baseline": baseline,
        "pairs": len(differences),
        "risk_difference": finite_or_none(mean(differences)),
        "ci95_low": finite_or_none(low),
        "ci95_high": finite_or_none(high),
        "baseline_only_failure": baseline_only_failure,
        "txnguard_only_failure": txnguard_only_failure,
        "mcnemar_exact_p": exact_mcnemar_pvalue(
            baseline_only_failure, txnguard_only_failure
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze_study(input_dirs: list[Path], output_dir: Path) -> dict[str, Any]:
    """Combine frozen blocks without modifying any immutable artifact."""
    if not input_dirs:
        raise ValueError("At least one input artifact is required.")
    for input_dir in input_dirs:
        if _inside(output_dir, input_dir):
            raise ValueError("Study-analysis output must be outside frozen input artifacts.")

    rows: list[dict[str, Any]] = []
    artifact_commits: dict[str, str | None] = {}
    for input_dir in input_dirs:
        manifest_path = input_dir / "artifact_manifest.json"
        if not manifest_path.exists():
            raise RuntimeError(f"Input artifact is not frozen: {input_dir}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifact = input_dir.name
        if artifact in artifact_commits:
            raise ValueError(f"Duplicate artifact name: {artifact}")
        artifact_commits[artifact] = manifest.get("code_commit")
        for row in load_summaries(input_dir):
            rows.append({"artifact": artifact, **row})
    if not rows:
        raise RuntimeError("No completed summaries found in input artifacts.")
    identities = [(row["artifact"], row["run_id"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise RuntimeError("Duplicate run IDs found within an input artifact.")

    output_dir.mkdir(parents=True, exist_ok=True)
    run_fields = ["artifact"] + sorted(set().union(*(row.keys() for row in rows)) - {"artifact"})
    with (output_dir / "combined_run_table.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=run_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    policies = {row["policy"] for row in rows}
    baselines = comparison_baselines(policies)
    paired_rows = [_paired_row(rows, baseline, {}) for baseline in baselines]
    _write_csv(output_dir / "paired_comparisons.csv", paired_rows)

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (row["artifact"], row.get("fault_mode", "none"), row["model_name"])
        ].append(row)
    stratified_rows = []
    for (artifact, fault_mode, model_name), items in sorted(grouped.items()):
        item_policies = {row["policy"] for row in items}
        for baseline in comparison_baselines(item_policies):
            stratified_rows.append(
                _paired_row(
                    items,
                    baseline,
                    {
                        "artifact": artifact,
                        "fault_mode": fault_mode,
                        "model_name": model_name,
                    },
                )
            )
    _write_csv(output_dir / "stratified_paired_comparisons.csv", stratified_rows)

    result = {
        "artifacts": artifact_commits,
        "runs": len(rows),
        "models": sorted({row["model_name"] for row in rows}),
        "policies": sorted(policies),
        "fault_modes": sorted({row.get("fault_mode", "none") for row in rows}),
        "paired_comparisons": paired_rows,
        "scope": "exploratory pilot; excluded from primary confirmatory estimates",
    }
    (output_dir / "study_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result
