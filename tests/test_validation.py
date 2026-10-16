import json

from txnagentbench.runner import run_matrix
from txnagentbench.benchmark import RunSpec
from txnagentbench.validation import artifact_run_id, validate_artifact


CONFIG = {
    "stage": "validation-test",
    "domains": ["inventory"],
    "policies": ["serial", "naive", "occ", "txnguard"],
    "agent_counts": [2],
    "contention_levels": [0, 100],
    "models": [{"name": "scripted-v1", "provider": "scripted"}],
    "scenario_seeds": [0],
    "schedule_seeds": [0],
    "fault_modes": ["none"],
}


def make_artifact(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(CONFIG), encoding="utf-8")
    output = tmp_path / "artifact"
    result = run_matrix(config_path, output, allow_scripted=True)
    assert result["completed"] == 8
    return output


def test_completed_artifact_validates(tmp_path):
    output = make_artifact(tmp_path)
    result = validate_artifact(output)
    assert result["valid"] is True
    assert result["checked_runs"] == 8


def test_corrupted_trace_is_rejected(tmp_path):
    output = make_artifact(tmp_path)
    trace_path = next((output / "runs").glob("*/trace.jsonl"))
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    events[-1]["summary"]["severe_violation"] = not events[-1]["summary"][
        "severe_violation"
    ]
    trace_path.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )
    result = validate_artifact(output)
    assert result["valid"] is False
    assert any("trace oracle does not match summary" in error for error in result["errors"])


def test_legacy_run_identity_omits_new_default_dimensions():
    spec = RunSpec("inventory", "serial", 2, 0, "scripted-v1", 0)
    legacy_config = {"domains": ["inventory"]}
    run_id, identity = artifact_run_id(spec, legacy_config)
    assert "scenario_seed" not in identity
    assert "fault_mode" not in identity
    assert run_id == "9144bd30b75e1a508d16"


def test_resume_rejects_changed_frozen_config(tmp_path):
    output = make_artifact(tmp_path)
    changed = dict(CONFIG)
    changed["schedule_seeds"] = [0, 1]
    changed_path = tmp_path / "changed.json"
    changed_path.write_text(json.dumps(changed), encoding="utf-8")
    try:
        run_matrix(changed_path, output, allow_scripted=True)
    except RuntimeError as exc:
        assert "different frozen configuration" in str(exc)
    else:
        raise AssertionError("Resume accepted a changed configuration")
