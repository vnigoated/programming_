import json

from pathlib import Path

from txnagentbench.readiness import primary_readiness, validate_primary_suite
from txnagentbench.faults import PRIMARY_BLOCK_FAULTS, PRIMARY_BLOCK_POLICIES


def test_primary_readiness_recognizes_implemented_publication_adapters(tmp_path):
    config = {
        "stage": "primary",
        "primary_design_id": "test-primary",
        "primary_block": "recovery",
        "domains": ["inventory"],
        "policies": sorted(PRIMARY_BLOCK_POLICIES["recovery"]),
        "agent_counts": [2],
        "contention_levels": [100],
        "models": [
            {
                "name": "locked-model-a",
                "provider": "transformers",
                "model_id": "example/model",
                "revision": "a" * 40,
                "quantization": "bf16",
                "prompt_version": "allocation-v2",
            },
            {
                "name": "locked-model-b",
                "provider": "transformers",
                "model_id": "example/model-b",
                "revision": "b" * 40,
                "quantization": "bf16",
                "prompt_version": "allocation-v2",
            },
        ],
        "scenario_seeds": [0],
        "schedule_seeds": [0],
        "fault_modes": sorted(PRIMARY_BLOCK_FAULTS["recovery"]),
    }
    config_path = tmp_path / "primary.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    result = primary_readiness(config_path, tmp_path / "persistent")
    checks = {row["check"]: row for row in result["checks"]}
    assert checks["persistent_output"]["passed"] is True
    assert checks["postgres_serializable_adapter"]["passed"] is True
    assert checks["mtpo_reproduction_adapter"]["passed"] is True
    assert checks["primary_block_fault_scope"]["passed"] is True


def test_checked_in_primary_suite_is_complete_and_locked():
    result = validate_primary_suite(
        [
            Path("configs/primary_v1_concurrency.json"),
            Path("configs/primary_v1_dependency.json"),
            Path("configs/primary_v1_recovery.json"),
        ]
    )
    assert result["ready"] is True
    assert result["planned_runs"] == {
        "concurrency": 6912,
        "dependency": 1728,
        "recovery": 6400,
    }
    assert result["total_planned_runs"] == 15040
