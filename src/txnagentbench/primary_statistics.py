from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from .analysis import bootstrap_interval, exact_mcnemar_pvalue, mean
from .runner import load_config
from .study_analysis import analyze_study


def holm_adjust(pvalues: list[float]) -> list[float]:
    indexed = sorted(enumerate(pvalues), key=lambda item: item[1])
    adjusted = [1.0] * len(pvalues)
    running = 0.0
    total = len(pvalues)
    for rank, (index, value) in enumerate(indexed):
        candidate = min(1.0, (total - rank) * value)
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _paired_bootstrap(values: list[float]) -> tuple[float, float]:
    """Efficient prespecified bootstrap for the {-1, 0, 1} paired differences."""
    if not values:
        return float("nan"), float("nan")
    try:
        import numpy as np
    except ImportError:
        return bootstrap_interval(values, samples=10000, seed=1701)
    support = np.array([-1.0, 0.0, 1.0])
    counts = np.array([values.count(value) for value in support], dtype=float)
    draws = np.random.default_rng(1701).multinomial(
        len(values), counts / len(values), size=10000
    )
    estimates = draws.dot(support) / len(values)
    low, high = np.quantile(estimates, [0.025, 0.975], method="inverted_cdf")
    return float(low), float(high)


def _pair_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["primary_block"],
        row["domain"],
        row["agent_count"],
        row["contention"],
        row["model_name"],
        row.get("model_revision"),
        row.get("scenario_seed", 0),
        row["schedule_seed"],
        row.get("fault_mode", "none"),
    )


def paired_comparison(
    rows: list[dict[str, Any]], specification: dict[str, Any]
) -> dict[str, Any]:
    selected = [row for row in rows if row["primary_block"] == specification["block"]]
    by_key: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in selected:
        by_key[_pair_key(row)][row["policy"]] = row
    treatment = specification["treatment"]
    comparator = specification["comparator"]
    outcome = specification["outcome"]
    differences: list[float] = []
    comparator_only_event = treatment_only_event = 0
    for policies in by_key.values():
        if treatment not in policies or comparator not in policies:
            continue
        treatment_event = bool(policies[treatment][outcome])
        comparator_event = bool(policies[comparator][outcome])
        differences.append(float(treatment_event) - float(comparator_event))
        comparator_only_event += int(comparator_event and not treatment_event)
        treatment_only_event += int(treatment_event and not comparator_event)
    low, high = _paired_bootstrap(differences)
    return {
        **specification,
        "pairs": len(differences),
        "risk_difference": mean(differences),
        "ci95_low": low,
        "ci95_high": high,
        "comparator_only_event": comparator_only_event,
        "treatment_only_event": treatment_only_event,
        "mcnemar_exact_p": exact_mcnemar_pvalue(
            comparator_only_event, treatment_only_event
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mixed_effects(rows: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        import pandas as pd
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    except ImportError as exc:
        return {"status": "unavailable", "reason": str(exc)}
    frame = pd.DataFrame(rows)
    frame["scenario_group"] = (
        frame["primary_block"].astype(str)
        + ":"
        + frame["domain"].astype(str)
        + ":"
        + frame["model_name"].astype(str)
        + ":"
        + frame["scenario_seed"].astype(str)
    )
    formula = (
        "severe_violation ~ C(policy, Treatment(reference='txnguard')) + C(domain) + "
        "C(model_name) + C(agent_count) + C(contention) + C(fault_mode)"
    )
    try:
        model = BinomialBayesMixedGLM.from_formula(
            formula, {"scenario": "0 + C(scenario_group)"}, frame
        )
        fitted = model.fit_vb()
    except Exception as exc:  # Statistical non-identifiability is itself reportable.
        return {"status": "not_estimable", "reason": f"{type(exc).__name__}: {exc}"}
    fixed_names = list(model.exog_names)
    fixed_means = [float(value) for value in fitted.params[: len(fixed_names)]]
    return {
        "status": "estimated",
        "method": "BinomialBayesMixedGLM variational Bayes",
        "formula": formula,
        "fixed_effect_log_odds": dict(zip(fixed_names, fixed_means)),
        "random_effect_log_sd_mean": [float(value) for value in fitted.vcp_mean],
    }


def analyze_primary(
    input_dirs: list[Path], output_dir: Path, plan_path: Path
) -> dict[str, Any]:
    plan = load_config(plan_path)
    rows: list[dict[str, Any]] = []
    design_ids = set()
    seen_blocks = set()
    for input_dir in input_dirs:
        if not (input_dir / "artifact_manifest.json").exists():
            raise RuntimeError(f"Primary input is not frozen: {input_dir}")
        config = load_config(input_dir / "frozen_config.json")
        if config.get("stage") != "primary":
            raise RuntimeError(f"Non-primary artifact supplied: {input_dir}")
        design_ids.add(config.get("primary_design_id"))
        block = config.get("primary_block")
        if block in seen_blocks:
            raise RuntimeError(f"Duplicate primary block: {block}")
        seen_blocks.add(block)
        for summary_path in sorted((input_dir / "runs").glob("*/summary.json")):
            rows.append(
                {
                    "primary_block": block,
                    **json.loads(summary_path.read_text(encoding="utf-8")),
                }
            )
    if design_ids != {plan["primary_design_id"]}:
        raise RuntimeError(f"Analysis plan and artifact design IDs differ: {design_ids}")
    if seen_blocks != {"concurrency", "dependency", "recovery"}:
        raise RuntimeError(f"Primary analysis requires all three blocks; found {seen_blocks}")

    output_dir.mkdir(parents=True, exist_ok=True)
    analyze_study(input_dirs, output_dir / "combined")
    confirmatory = [paired_comparison(rows, item) for item in plan["confirmatory"]]
    secondary = [paired_comparison(rows, item) for item in plan["secondary"]]
    adjusted = holm_adjust([row["mcnemar_exact_p"] for row in secondary])
    for row, pvalue in zip(secondary, adjusted):
        row["holm_adjusted_p"] = pvalue
        row["reject_at_alpha"] = pvalue < float(plan["alpha"])
    noninferiority = [paired_comparison(rows, item) for item in plan["noninferiority"]]
    margin = float(plan["noninferiority_margin"])
    for row in noninferiority:
        row["margin"] = margin
        row["noninferior"] = row["ci95_low"] > -margin

    _write_csv(output_dir / "confirmatory.csv", confirmatory)
    _write_csv(output_dir / "secondary_holm.csv", secondary)
    _write_csv(output_dir / "noninferiority.csv", noninferiority)
    mixed = _mixed_effects(rows)
    (output_dir / "mixed_effects.json").write_text(
        json.dumps(mixed, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    result = {
        "primary_design_id": plan["primary_design_id"],
        "runs": len(rows),
        "confirmatory": confirmatory,
        "secondary": secondary,
        "noninferiority": noninferiority,
        "mixed_effects": mixed,
    }
    (output_dir / "primary_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result
