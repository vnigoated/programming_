from txnagentbench.primary_statistics import holm_adjust, paired_comparison


def test_holm_adjustment_is_monotone_in_sorted_order():
    assert holm_adjust([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_primary_paired_comparison_uses_treatment_minus_comparator():
    common = {
        "primary_block": "concurrency",
        "domain": "inventory",
        "agent_count": 2,
        "contention": 100,
        "model_name": "model-a",
        "model_revision": "a" * 40,
        "scenario_seed": 0,
        "schedule_seed": 0,
        "fault_mode": "none",
    }
    rows = [
        {**common, "policy": "txnguard", "severe_violation": False},
        {**common, "policy": "naive", "severe_violation": True},
    ]
    result = paired_comparison(
        rows,
        {
            "name": "test",
            "block": "concurrency",
            "outcome": "severe_violation",
            "treatment": "txnguard",
            "comparator": "naive",
        },
    )
    assert result["pairs"] == 1
    assert result["risk_difference"] == -1.0
    assert result["comparator_only_event"] == 1
    assert result["treatment_only_event"] == 0
