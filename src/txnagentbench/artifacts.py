from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .runner import atomic_json, load_config
from .validation import validate_artifact


MANIFEST_NAME = "artifact_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(output_dir: Path) -> dict[str, Any]:
    validation = validate_artifact(output_dir)
    if not validation["valid"]:
        details = "; ".join(validation["errors"][:5])
        raise RuntimeError(f"Cannot freeze an invalid artifact: {details}")

    config = load_config(output_dir / "frozen_config.json")
    environment = load_config(output_dir / "environment.json")
    files = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == MANIFEST_NAME or path.name.endswith(".tmp"):
            continue
        relative = path.relative_to(output_dir).as_posix()
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    model_locks = [
        {
            "name": model.get("name"),
            "model_id": model.get("model_id"),
            "revision": model.get("revision"),
            "quantization": model.get("quantization"),
            "prompt_version": model.get("prompt_version"),
        }
        for model in config.get("models", [])
    ]
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_root": output_dir.name,
        "stage": config.get("stage"),
        "code_commit": environment.get("git_commit"),
        "git_dirty": environment.get("git_dirty"),
        "frozen_config_sha256": sha256_file(output_dir / "frozen_config.json"),
        "models": model_locks,
        "validation": {
            "expected_runs": validation["expected_runs"],
            "completed_runs": validation["completed_runs"],
            "checked_runs": validation["checked_runs"],
        },
        "file_count": len(files),
        "files": files,
    }


def freeze_artifact(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / MANIFEST_NAME
    if manifest_path.exists():
        raise RuntimeError(
            f"{MANIFEST_NAME} already exists; frozen artifacts are immutable. "
            "Use a new output directory for a new artifact."
        )
    manifest = build_manifest(output_dir)
    atomic_json(manifest_path, manifest)
    return {
        "frozen": True,
        "manifest": str(manifest_path),
        "files_hashed": manifest["file_count"],
        "code_commit": manifest["code_commit"],
        "model_revisions": [model["revision"] for model in manifest["models"]],
    }


def verify_manifest(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / MANIFEST_NAME
    if not manifest_path.exists():
        return {"valid": False, "errors": [f"missing {MANIFEST_NAME}"]}
    manifest = load_config(manifest_path)
    errors: list[str] = []
    expected_paths = set()
    entries = manifest.get("files", [])
    if not isinstance(entries, list):
        return {"valid": False, "errors": ["manifest files field is not a list"]}
    if manifest.get("file_count") != len(entries):
        errors.append("manifest file_count does not match files list")
    for entry in entries:
        relative = entry.get("path")
        if not isinstance(relative, str):
            errors.append("manifest contains an invalid path")
            continue
        if relative in expected_paths:
            errors.append(f"duplicate manifest path: {relative}")
        expected_paths.add(relative)
        path = output_dir / Path(relative)
        try:
            resolved = path.resolve()
            resolved.relative_to(output_dir.resolve())
        except (OSError, ValueError):
            errors.append(f"unsafe manifest path: {relative}")
            continue
        if not path.is_file():
            errors.append(f"missing file: {relative}")
            continue
        if path.stat().st_size != entry.get("bytes"):
            errors.append(f"size mismatch: {relative}")
        if sha256_file(path) != entry.get("sha256"):
            errors.append(f"hash mismatch: {relative}")

    actual_paths = {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != MANIFEST_NAME and not path.name.endswith(".tmp")
    }
    for relative in sorted(actual_paths - expected_paths):
        errors.append(f"untracked file: {relative}")
    for relative in sorted(expected_paths - actual_paths):
        if not any(error == f"missing file: {relative}" for error in errors):
            errors.append(f"missing file: {relative}")

    config_path = output_dir / "frozen_config.json"
    environment_path = output_dir / "environment.json"
    if config_path.is_file() and sha256_file(config_path) != manifest.get("frozen_config_sha256"):
        errors.append("frozen_config_sha256 field does not match frozen_config.json")
    if environment_path.is_file():
        environment = load_config(environment_path)
        if environment.get("git_commit") != manifest.get("code_commit"):
            errors.append("code_commit does not match environment.json")
    return {
        "valid": not errors,
        "files_checked": len(expected_paths),
        "errors": errors,
    }

