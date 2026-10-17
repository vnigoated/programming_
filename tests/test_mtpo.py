import pytest

from txnagentbench.benchmark import RunSpec, run_benchmark
from txnagentbench.models import ScriptedDecisionModel


@pytest.mark.parametrize("domain", ["inventory", "calendar", "support", "cloud"])
@pytest.mark.parametrize("scenario_seed", [0, 1, 2, 3])
@pytest.mark.parametrize(
    "fault_mode",
    [
        "none",
        "duplicate_effect",
        "dependency_change",
        "crash_before_commit",
        "crash_after_commit_before_ack",
        "delayed_ack",
        "compensation_failure",
    ],
)
def test_mtpo_reproduction_is_safe(domain, scenario_seed, fault_mode):
    spec = RunSpec(
        domain,
        "mtpo_reproduction",
        4,
        100,
        "scripted-v1",
        schedule_seed=1,
        scenario_seed=scenario_seed,
        fault_mode=fault_mode,
    )
    summary, trace = run_benchmark(spec, ScriptedDecisionModel())
    assert summary["task_success"] is True
    assert summary["severe_violation"] is False
    assert summary["baseline_provenance"]["official_author_code"] is False
    assert any(event.get("event") == "trajectory_reconcile" for event in trace)


@pytest.mark.parametrize(
    "fault_mode",
    [
        "crash_before_commit",
        "crash_after_commit_before_ack",
        "delayed_ack",
        "compensation_failure",
    ],
)
def test_mtpo_reproduction_records_successful_recovery(fault_mode):
    spec = RunSpec(
        "inventory",
        "mtpo_reproduction",
        2,
        0,
        "scripted-v1",
        schedule_seed=0,
        scenario_seed=0,
        fault_mode=fault_mode,
    )
    summary, _ = run_benchmark(spec, ScriptedDecisionModel())
    assert summary["recovery_attempts"] > 0
    assert summary["recovery_rate"] == 1.0
    assert summary["recovery_violation"] is False


def test_mtpo_notifies_and_repairs_stale_shared_premise():
    spec = RunSpec(
        "inventory", "mtpo_reproduction", 2, 100, "scripted-v1", 1, scenario_seed=0
    )
    summary, _ = run_benchmark(spec, ScriptedDecisionModel())
    assert summary["mtpo_notifications"] >= 1
    assert summary["semantic_repairs"] >= 1
