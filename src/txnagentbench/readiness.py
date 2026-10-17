from __future__ import annotations

from pathlib import Path
import hashlib
import json
import tempfile
from typing import Any

from .benchmark import IMPLEMENTED_POLICIES, POLICY_UNSUPPORTED_FAULTS
from .faults import PRIMARY_BLOCK_FAULTS, PRIMARY_BLOCK_POLICIES, PRIMARY_FAULT_MODES
from .runner import (
    IMMUTABLE_REVISION,
    environment_manifest,
    expand_matrix,
    load_config,
    validate_config,
)


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
    block = config.get("primary_block")
    expected_policies = PRIMARY_BLOCK_POLICIES.get(block, REQUIRED_PRIMARY_POLICIES)
    missing = sorted(expected_policies - policies)
    unexpected = sorted(policies - expected_policies)
    add(
        "required_policy_names",
        not missing and not unexpected,
        "The block contains exactly its prespecified policies."
        if not missing and not unexpected
        else f"Missing: {missing}; unexpected: {unexpected}",
    )
    expected_faults = PRIMARY_BLOCK_FAULTS.get(block, frozenset())
    observed_faults = set(config.get("fault_modes", ["none"]))
    fault_scope_ready = bool(expected_faults) and observed_faults == expected_faults
    add(
        "primary_block_fault_scope",
        fault_scope_ready,
        f"block={block!r}, faults={sorted(observed_faults)}",
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
        "note": "Readiness is environmental and must be rerun against the final frozen config.",
    }


def validate_primary_suite(config_paths: list[Path]) -> dict[str, Any]:
    errors: list[str] = []
    configs: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for path in config_paths:
        try:
            config = load_config(path)
            validate_config(config)
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
            errors.append(f"{path}: {exc}")
            continue
        block = config["primary_block"]
        if block in configs:
            errors.append(f"Duplicate primary block: {block}")
            continue
        configs[block] = config
        hashes[block] = hashlib.sha256(path.read_bytes()).hexdigest()

    missing_blocks = sorted(set(PRIMARY_BLOCK_FAULTS) - set(configs))
    if missing_blocks:
        errors.append(f"Missing primary blocks: {missing_blocks}")
    unexpected_blocks = sorted(set(configs) - set(PRIMARY_BLOCK_FAULTS))
    if unexpected_blocks:
        errors.append(f"Unexpected primary blocks: {unexpected_blocks}")

    design_ids = {config.get("primary_design_id") for config in configs.values()}
    if len(design_ids) != 1:
        errors.append(f"Primary design IDs differ: {sorted(str(x) for x in design_ids)}")
    model_locks = {
        json.dumps(config.get("models", []), sort_keys=True)
        for config in configs.values()
    }
    if len(model_locks) != 1:
        errors.append("Primary blocks do not share identical model locks.")
    observed_faults = set().union(
        *(set(config.get("fault_modes", [])) for config in configs.values())
    )
    if observed_faults != PRIMARY_FAULT_MODES:
        errors.append(
            f"Primary suite fault union differs: expected {sorted(PRIMARY_FAULT_MODES)}, "
            f"found {sorted(observed_faults)}"
        )

    planned_runs = {
        block: len(expand_matrix(config)) for block, config in configs.items()
    }
    return {
        "ready": not errors,
        "primary_design_id": next(iter(design_ids)) if len(design_ids) == 1 else None,
        "config_sha256": hashes,
        "planned_runs": planned_runs,
        "total_planned_runs": sum(planned_runs.values()),
        "errors": errors,
    }

