from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .runner import expand_matrix, load_config


def artifact_run_id(spec, config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    identity = asdict(spec)
    if "scenario_seeds" not in config:
        identity.pop("scenario_seed")
    if "fault_modes" not in config:
        identity.pop("fault_mode")
    raw = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:20], identity


def validate_artifact(output_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    config_path = output_dir / "frozen_config.json"
    environment_path = output_dir / "environment.json"
    if not config_path.exists():
        return {"valid": False, "errors": ["missing frozen_config.json"]}
    if not environment_path.exists():
        errors.append("missing environment.json")

    config = load_config(config_path)
    expected_rows = expand_matrix(config)
    expected = {}
    for spec, _ in expected_rows:
        run_id, identity = artifact_run_id(spec, config)
        expected[run_id] = identity
    run_root = output_dir / "runs"
    actual_summary_paths = sorted(run_root.glob("*/summary.json"))
    actual = {path.parent.name: path for path in actual_summary_paths}
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    if missing:
        errors.append(f"missing completed runs: {len(missing)}")
    if unexpected:
        errors.append(f"unexpected completed runs: {len(unexpected)}")

    checked = 0
    for run_id in sorted(set(expected) & set(actual)):
        summary_path = actual[run_id]
        trace_path = summary_path.parent / "trace.jsonl"
        error_path = summary_path.parent / "error.json"
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{run_id}: unreadable summary ({exc})")
            continue
        if summary.get("run_id") != run_id:
            errors.append(f"{run_id}: summary run_id mismatch")
        expected_fields = expected[run_id]
        for field, value in expected_fields.items():
            if summary.get(field) != value:
                errors.append(f"{run_id}: summary field {field} does not match frozen config")
        if summary.get("completed") is not True:
            errors.append(f"{run_id}: summary is not marked completed")
        if error_path.exists():
            errors.append(f"{run_id}: both summary.json and error.json exist")
        if not trace_path.exists():
            errors.append(f"{run_id}: missing trace.jsonl")
            continue
        try:
            events = [
                json.loads(line)
                for line in trace_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{run_id}: unreadable trace ({exc})")
            continue
        if not events:
            errors.append(f"{run_id}: empty trace")
            continue
        if not any(event.get("event") == "read" for event in events):
            errors.append(f"{run_id}: trace has no read event")
        if not any(event.get("event") == "decision" for event in events):
            errors.append(f"{run_id}: trace has no decision event")
        final = events[-1]
        if final.get("event") != "oracle":
            errors.append(f"{run_id}: final trace event is not oracle")
        elif final.get("summary") != summary:
            errors.append(f"{run_id}: trace oracle does not match summary")
        checked += 1

    return {
        "valid": not errors,
        "expected_runs": len(expected),
        "completed_runs": len(actual),
        "checked_runs": checked,
        "missing_run_ids": missing,
        "unexpected_run_ids": unexpected,
        "errors": errors,
    }
