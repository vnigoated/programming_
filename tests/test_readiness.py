import json

from txnagentbench.readiness import primary_readiness


def test_primary_readiness_recognizes_implemented_publication_adapters(tmp_path):
    config = {
        "stage": "primary",
        "domains": ["inventory"],
        "policies": ["naive", "postgres_serializable", "mtpo_reproduction", "txnguard"],
        "agent_counts": [2],
        "contention_levels": [100],
        "models": [
            {
                "name": "locked-model",
                "provider": "transformers",
                "model_id": "example/model",
                "revision": "a" * 40,
                "quantization": "bf16",
                "prompt_version": "allocation-v2",
            }
        ],
        "scenario_seeds": [0],
        "schedule_seeds": [0],
        "fault_modes": ["none"],
    }
    config_path = tmp_path / "primary.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    result = primary_readiness(config_path, tmp_path / "persistent")
    checks = {row["check"]: row for row in result["checks"]}
    assert result["ready"] is False
    assert checks["persistent_output"]["passed"] is True
    assert checks["postgres_serializable_adapter"]["passed"] is True
    assert checks["mtpo_reproduction_adapter"]["passed"] is True
