from pathlib import Path

from txnagentbench.runner import expand_matrix, load_config


def test_recovery_ablation_smoke_covers_both_mechanism_axes():
    config = load_config(Path("configs/recovery_ablation_smoke.json"))
    rows = expand_matrix(config)
    assert len(rows) == 240
    assert {
        "serial_recovery",
        "recovery_only",
        "occ_recovery",
        "txnguard_no_recovery",
        "txnguard",
    } == set(config["policies"])


def test_mistral_smoke_is_small_and_immutably_pinned():
    config = load_config(Path("configs/diversity_model_smoke_mistral7b.json"))
    rows = expand_matrix(config)
    assert len(rows) == 16
    model = config["models"][0]
    assert model["model_id"] == "mistralai/Mistral-7B-Instruct-v0.3"
    assert model["revision"] == "c170c708c41dac9275d15a8fff4eca08d52bab71"
