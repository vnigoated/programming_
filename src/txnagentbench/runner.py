from __future__ import annotations

import itertools
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

from .benchmark import RunSpec, run_benchmark
from .models import build_model


IMMUTABLE_REVISION = re.compile(r"[0-9a-fA-F]{40}")


def validate_config(config: dict[str, Any]) -> None:
    required = {
        "stage",
        "domains",
        "policies",
        "agent_counts",
        "contention_levels",
        "models",
        "schedule_seeds",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing configuration fields: {', '.join(missing)}")
    if not config["models"]:
        raise ValueError("At least one model must be configured.")
    if any(int(value) < 0 or int(value) > 100 for value in config["contention_levels"]):
        raise ValueError("Contention levels must be between 0 and 100.")
    if any(int(value) < 1 for value in config["agent_counts"]):
        raise ValueError("Agent counts must be positive.")
    unknown_faults = sorted(set(config.get("fault_modes", ["none"])) - {"none", "duplicate_effect"})
    if unknown_faults:
        raise ValueError(f"Unknown fault modes: {', '.join(unknown_faults)}")
    if config["stage"] in {"model-smoke", "model_smoke", "pilot", "primary"}:
        for model in config["models"]:
            if model.get("provider") == "scripted":
                continue
            revision = str(model.get("revision", ""))
            if not IMMUTABLE_REVISION.fullmatch(revision):
                raise ValueError(
                    f"Model {model.get('name', '<unnamed>')} must use a 40-character immutable revision."
                )


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def expand_matrix(config: dict[str, Any]) -> list[tuple[RunSpec, dict[str, Any]]]:
    validate_config(config)
    rows: list[tuple[RunSpec, dict[str, Any]]] = []
    for (
        domain,
        policy,
        agents,
        contention,
        model,
        scenario_seed,
        schedule_seed,
        fault_mode,
    ) in itertools.product(
        config["domains"],
        config["policies"],
        config["agent_counts"],
        config["contention_levels"],
        config["models"],
        config.get("scenario_seeds", [0]),
        config["schedule_seeds"],
        config.get("fault_modes", ["none"]),
    ):
        spec = RunSpec(
            domain=domain,
            policy=policy,
            agent_count=int(agents),
            contention=int(contention),
            model_name=model["name"],
            schedule_seed=int(schedule_seed),
            scenario_seed=int(scenario_seed),
            fault_mode=str(fault_mode),
        )
        rows.append((spec, model))
    return sorted(rows, key=lambda item: (item[0].model_name, item[0].run_id))


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def environment_manifest() -> dict[str, Any]:
    manifest = {
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": git_commit(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        manifest["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        manifest["nvidia_smi"] = "unavailable"
    return manifest


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_trace(path: Path, events: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run_matrix(
    config_path: Path,
    output_dir: Path,
    max_runs: int | None = None,
    allow_scripted: bool = False,
) -> dict[str, int]:
    config = load_config(config_path)
    rows = expand_matrix(config)
    if config.get("stage") == "primary":
        if any(model.get("provider") == "scripted" for model in config["models"]):
            raise RuntimeError("Primary experiments cannot use scripted agents.")
        if "coagent" not in config["policies"]:
            raise RuntimeError("Primary config must include CoAgent or document an approved protocol change.")
    elif not allow_scripted and any(model.get("provider") == "scripted" for model in config["models"]):
        raise RuntimeError("Pass --allow-scripted for engineering-only smoke tests.")

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "environment.json", environment_manifest())
    atomic_json(output_dir / "frozen_config.json", config)
    counts = {"planned": len(rows), "completed": 0, "skipped": 0, "failed": 0}
    current_name: str | None = None
    model_instance = None
    attempted = 0
    try:
        for spec, model_config in rows:
            summary_path = output_dir / "runs" / spec.run_id / "summary.json"
            if summary_path.exists():
                counts["skipped"] += 1
                continue
            if max_runs is not None and attempted >= max_runs:
                break
            attempted += 1
            if current_name != spec.model_name:
                if model_instance is not None:
                    model_instance.close()
                model_instance = build_model(model_config)
                current_name = spec.model_name
            run_dir = summary_path.parent
            try:
                summary, trace = run_benchmark(spec, model_instance)
                write_trace(run_dir / "trace.jsonl", trace)
                atomic_json(summary_path, summary)
                counts["completed"] += 1
            except Exception as exc:
                atomic_json(
                    run_dir / "error.json",
                    {"run_id": spec.run_id, "type": type(exc).__name__, "message": str(exc)},
                )
                counts["failed"] += 1
                if isinstance(exc, NotImplementedError):
                    raise
    finally:
        if model_instance is not None:
            model_instance.close()
    atomic_json(output_dir / "progress.json", counts)
    return counts

