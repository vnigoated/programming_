import json

import pytest

from txnagentbench.artifacts import freeze_artifact, verify_manifest
from txnagentbench.runner import run_matrix


CONFIG = {
    "stage": "artifact-test",
    "domains": ["inventory"],
    "policies": ["serial"],
    "agent_counts": [2],
    "contention_levels": [100],
    "models": [{"name": "scripted-v1", "provider": "scripted"}],
    "scenario_seeds": [0],
    "schedule_seeds": [0],
    "fault_modes": ["none"],
}


def make_artifact(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(CONFIG), encoding="utf-8")
    output = tmp_path / "artifact"
    run_matrix(config_path, output, allow_scripted=True)
    return output


def test_freeze_and_verify(tmp_path):
    output = make_artifact(tmp_path)
    result = freeze_artifact(output)
    assert result["frozen"] is True
    assert result["files_hashed"] >= 4
    assert verify_manifest(output)["valid"] is True


def test_manifest_detects_tampering(tmp_path):
    output = make_artifact(tmp_path)
    freeze_artifact(output)
    summary = next((output / "runs").glob("*/summary.json"))
    summary.write_text(summary.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    result = verify_manifest(output)
    assert result["valid"] is False
    assert any("mismatch" in error for error in result["errors"])


def test_freeze_refuses_overwrite(tmp_path):
    output = make_artifact(tmp_path)
    freeze_artifact(output)
    with pytest.raises(RuntimeError, match="already exists"):
        freeze_artifact(output)


def test_runner_refuses_to_modify_frozen_artifact(tmp_path):
    output = make_artifact(tmp_path)
    freeze_artifact(output)
    config_path = tmp_path / "config.json"
    with pytest.raises(RuntimeError, match="artifact is frozen"):
        run_matrix(config_path, output, allow_scripted=True)
