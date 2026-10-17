from pathlib import Path
import json

import pytest

from txnagentbench.runner import expand_matrix, load_config, validate_config


CONFIGS = {
    "primary_v1_concurrency.json": ("concurrency", 6912),
    "primary_v1_dependency.json": ("dependency", 1728),
    "primary_v1_recovery.json": ("recovery", 6400),
}


@pytest.mark.parametrize("filename,expected", CONFIGS.items())
def test_primary_block_counts_and_identity(filename, expected):
    block, count = expected
    config = load_config(Path("configs") / filename)
    validate_config(config)
    assert config["primary_design_id"] == "txnagentbench-primary-v1"
    assert config["primary_block"] == block
    assert len(expand_matrix(config)) == count


def test_primary_config_rejects_incomplete_fault_scope():
    config = load_config(Path("configs/primary_v1_recovery.json"))
    config["fault_modes"] = ["duplicate_effect"]
    with pytest.raises(ValueError, match="must use exactly"):
        validate_config(config)


def test_primary_config_rejects_removed_control():
    config = load_config(Path("configs/primary_v1_recovery.json"))
    config["policies"].remove("occ_recovery")
    with pytest.raises(ValueError, match="must use exactly"):
        validate_config(config)


def test_primary_analysis_and_oracle_plans_share_design_id():
    analysis = json.loads(
        Path("configs/primary_v1_analysis_plan.json").read_text(encoding="utf-8")
    )
    audit = json.loads(
        Path("configs/primary_v1_oracle_audit_plan.json").read_text(encoding="utf-8")
    )
    assert analysis["primary_design_id"] == "txnagentbench-primary-v1"
    assert audit["primary_design_id"] == "txnagentbench-primary-v1"
    assert analysis["noninferiority_margin"] == 0.05
    assert audit["sample_size"] == 240
