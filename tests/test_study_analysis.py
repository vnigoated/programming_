import json

import pytest

from txnagentbench.study_analysis import analyze_study


def write_artifact(root, name, rows):
    artifact = root / name
    (artifact / "runs").mkdir(parents=True)
    (artifact / "artifact_manifest.json").write_text(
        json.dumps({"code_commit": "a" * 40}), encoding="utf-8"
    )
    for row in rows:
        run_dir = artifact / "runs" / row["run_id"]
        run_dir.mkdir()
        (run_dir / "summary.json").write_text(json.dumps(row), encoding="utf-8")
    return artifact


def row(run_id, policy, severe):
    return {
        "run_id": run_id,
        "domain": "inventory",
        "policy": policy,
        "agent_count": 2,
        "contention": 100,
        "model_name": "model-a",
        "schedule_seed": 0,
        "scenario_seed": 0,
        "fault_mode": "none",
        "severe_violation": severe,
    }


def test_study_analysis_includes_every_baseline_and_preserves_inputs(tmp_path):
    artifact = write_artifact(
        tmp_path,
        "core",
        [
            row("guard", "txnguard", False),
            row("naive", "naive", True),
            row("postgres", "postgres_serializable", True),
            row("mtpo", "mtpo_reproduction", False),
        ],
    )
    before = sorted(path.relative_to(artifact) for path in artifact.rglob("*"))
    result = analyze_study([artifact], tmp_path / "analysis")
    after = sorted(path.relative_to(artifact) for path in artifact.rglob("*"))

    assert result["runs"] == 4
    assert [item["baseline"] for item in result["paired_comparisons"]] == [
        "naive",
        "postgres_serializable",
        "mtpo_reproduction",
    ]
    assert before == after


def test_study_analysis_refuses_to_write_inside_frozen_artifact(tmp_path):
    artifact = write_artifact(tmp_path, "core", [row("guard", "txnguard", False)])
    with pytest.raises(ValueError, match="outside frozen"):
        analyze_study([artifact], artifact / "new-analysis")
