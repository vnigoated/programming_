import pytest

from txnagentbench.benchmark import RunSpec, run_benchmark
from txnagentbench.models import ScriptedDecisionModel


DOMAINS = ["inventory", "calendar", "support", "cloud"]
RECOVERY_FAULTS = [
    "crash_before_commit",
    "crash_after_commit_before_ack",
    "delayed_ack",
    "compensation_failure",
]


def run_fault(domain: str, policy: str, fault_mode: str):
    spec = RunSpec(
        domain,
        policy,
        2,
        0,
        "scripted-v1",
        schedule_seed=0,
        scenario_seed=0,
        fault_mode=fault_mode,
    )
    return run_benchmark(spec, ScriptedDecisionModel())


@pytest.mark.parametrize("domain", DOMAINS)
@pytest.mark.parametrize("fault_mode", RECOVERY_FAULTS)
def test_txnguard_recovers_workflow_faults(domain, fault_mode):
    summary, trace = run_fault(domain, "txnguard", fault_mode)
    assert summary["task_success"] is True
    assert summary["severe_violation"] is False
    assert summary["recovery_violation"] is False
    assert summary["recovery_attempts"] > 0
    assert summary["recovery_rate"] == 1.0
    assert any(
        event.get("event")
        in {"worker_crash", "ack_timeout", "compensation"}
        for event in trace
    )


@pytest.mark.parametrize("domain", DOMAINS)
@pytest.mark.parametrize("policy", ["serial_recovery", "occ_recovery", "recovery_only"])
@pytest.mark.parametrize("fault_mode", RECOVERY_FAULTS)
def test_recovery_capability_controls_recover_workflow_faults(
    domain, policy, fault_mode
):
    summary, _ = run_fault(domain, policy, fault_mode)
    assert summary["task_success"] is True
    assert summary["severe_violation"] is False
    assert summary["recovery_violation"] is False
    assert summary["recovery_attempts"] > 0
    assert summary["recovery_rate"] == 1.0


@pytest.mark.parametrize("domain", DOMAINS)
@pytest.mark.parametrize("policy", ["serial", "naive", "occ", "txnguard_no_recovery"])
@pytest.mark.parametrize("fault_mode", RECOVERY_FAULTS)
def test_transaction_only_policies_do_not_recover_workflow_faults(
    domain, policy, fault_mode
):
    summary, _ = run_fault(domain, policy, fault_mode)
    assert summary["task_success"] is False
    assert summary["severe_violation"] is True
    assert summary["recovery_violation"] is True
    assert summary["recovery_attempts"] > 0
    assert summary["recovery_rate"] == 0.0


@pytest.mark.parametrize("domain", DOMAINS)
def test_successful_compensation_removes_committed_effect(domain):
    summary, _ = run_fault(domain, "txnguard", "compensation_failure")
    assert summary["accepted"] == 0
    assert summary["rejected"] == 2
    assert summary["side_effect_count"] == 0


def test_recovery_only_does_not_hide_concurrency_failures():
    spec = RunSpec("inventory", "recovery_only", 2, 100, "scripted-v1", 0)
    summary, _ = run_benchmark(spec, ScriptedDecisionModel())
    assert summary["lost_update"] is True
    assert summary["severe_violation"] is True


def test_occ_recovery_combines_version_validation_and_recovery():
    spec = RunSpec(
        "inventory",
        "occ_recovery",
        2,
        100,
        "scripted-v1",
        0,
        fault_mode="duplicate_effect",
    )
    summary, _ = run_benchmark(spec, ScriptedDecisionModel())
    assert summary["task_success"] is True
    assert summary["severe_violation"] is False
    assert summary["conflicts"] >= 1
    assert summary["recovery_rate"] == 1.0
