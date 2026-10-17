from pathlib import Path

from txnagentbench.runner import expand_matrix, load_config


BLOCKS = {
    "expanded_pilot_core_qwen25_7b.json": (1536, {"none", "duplicate_effect"}),
    "expanded_pilot_dependency_qwen25_7b.json": (192, {"dependency_change"}),
    "expanded_pilot_recovery_qwen25_7b.json": (
        1536,
        {
            "crash_before_commit",
            "crash_after_commit_before_ack",
            "delayed_ack",
            "compensation_failure",
        },
    ),
}


def test_expanded_pilot_blocks_are_disjoint_and_frozen():
    observed_faults = set()
    total_runs = 0
    for filename, (expected_runs, expected_faults) in BLOCKS.items():
        config = load_config(Path("configs") / filename)
        rows = expand_matrix(config)
        assert config["stage"] == "pilot"
        assert len(rows) == expected_runs
        assert set(config["fault_modes"]) == expected_faults
        assert not observed_faults & expected_faults
        assert "postgres_serializable" in config["policies"]
        assert config["models"][0]["revision"] == (
            "a09a35458c702b33eeacc393d103063234e8bc28"
        )
        observed_faults.update(expected_faults)
        total_runs += len(rows)

    assert total_runs == 3264
