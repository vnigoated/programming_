from txnagentbench.benchmark import RunSpec, run_benchmark
from txnagentbench.models import ScriptedDecisionModel


def run(policy: str, contention: int = 100):
    spec = RunSpec("inventory", policy, 2, contention, "scripted-v1", 0)
    return run_benchmark(spec, ScriptedDecisionModel())[0]


def test_naive_parallel_exposes_lost_update():
    result = run("naive")
    assert result["lost_update"] is True
    assert result["severe_violation"] is True


def test_serial_is_safe():
    result = run("serial")
    assert result["severe_violation"] is False
    assert result["task_success"] is True


def test_txnguard_is_safe_under_contention():
    result = run("txnguard")
    assert result["severe_violation"] is False
    assert result["conflicts"] >= 1


def test_zero_contention_is_safe_for_naive_policy():
    result = run("naive", contention=0)
    assert result["severe_violation"] is False

