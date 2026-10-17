from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any

from .benchmark import IMPLEMENTED_POLICIES, POLICY_UNSUPPORTED_FAULTS
from .runner import IMMUTABLE_REVISION, environment_manifest, load_config, validate_config


REQUIRED_PRIMARY_POLICIES = {
    "naive",
    "postgres_serializable",
    "mtpo_reproduction",
    "txnguard",
}


def _persistent_write_probe(output_dir: Path | None) -> tuple[bool, str]:
    if output_dir is None:
        return False, "Pass --output with a mounted persistent artifact directory."
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output_dir, prefix=".txnagentbench-probe-", delete=False
        ) as handle:
            probe = Path(handle.name)
            handle.write("persistent-write-ok\n")
            handle.flush()
        contents = probe.read_text(encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return False, f"Write probe failed: {exc}"
    return contents == "persistent-write-ok\n", f"Write probe passed in {output_dir.resolve()}"


def primary_readiness(config_path: Path, output_dir: Path | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": passed, "detail": detail})

    try:
        validate_config(config)
    except (ValueError, TypeError) as exc:
        add("configuration_schema", False, str(exc))
    else:
        add("configuration_schema", True, "Required fields and locked model metadata are valid.")

    add(
        "primary_stage",
        config.get("stage") == "primary",
        f"stage={config.get('stage')!r}",
    )
    scripted = [m.get("name") for m in config.get("models", []) if m.get("provider") == "scripted"]
    add("real_models", not scripted, "No scripted models." if not scripted else f"Scripted: {scripted}")
    mutable = [
        m.get("name")
        for m in config.get("models", [])
        if not IMMUTABLE_REVISION.fullmatch(str(m.get("revision", "")))
    ]
    add(
        "immutable_model_revisions",
        not mutable,
        "All revisions are 40-character commits." if not mutable else f"Unpinned: {mutable}",
    )
    policies = set(config.get("policies", []))
    missing = sorted(REQUIRED_PRIMARY_POLICIES - policies)
    add(
        "required_policy_names",
        not missing,
        "All primary policy names are present." if not missing else f"Missing: {missing}",
    )

    unsupported_postgres_faults = sorted(
        set(config.get("fault_modes", ["none"]))
        & POLICY_UNSUPPORTED_FAULTS.get("postgres_serializable", set())
    )
    postgres_ready = (
        "postgres_serializable" in IMPLEMENTED_POLICIES and not unsupported_postgres_faults
    )
    mtpo_ready = "mtpo_reproduction" in IMPLEMENTED_POLICIES
    add(
        "postgres_serializable_adapter",
        postgres_ready,
        "Implemented for all configured faults."
        if postgres_ready
        else (
            f"Unsupported configured faults: {unsupported_postgres_faults}"
            if unsupported_postgres_faults
            else "Not implemented: real PostgreSQL transactions are required."
        ),
    )
    add(
        "mtpo_reproduction_adapter",
        mtpo_ready,
        "Implemented."
        if mtpo_ready
        else "Not implemented: independent paper-specified MTPO reproduction is required.",
    )

    environment = environment_manifest()
    add(
        "clean_git_worktree",
        environment.get("git_dirty") is False,
        f"git_commit={environment.get('git_commit')}, git_dirty={environment.get('git_dirty')}",
    )
    output_ready, output_detail = _persistent_write_probe(output_dir)
    add("persistent_output", output_ready, output_detail)

    blockers = [check for check in checks if not check["passed"]]
    return {
        "ready": not blockers,
        "checks": checks,
        "blockers": blockers,
        "note": "A false readiness result is expected until publication baselines are implemented.",
    }

