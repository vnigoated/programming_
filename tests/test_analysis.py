import json

from txnagentbench.analysis import exact_mcnemar_pvalue, result_scope_note


def write_config(tmp_path, stage, provider):
    (tmp_path / "frozen_config.json").write_text(
        json.dumps({"stage": stage, "models": [{"provider": provider}]}),
        encoding="utf-8",
    )


def test_real_model_smoke_is_labeled_as_integration_evidence(tmp_path):
    write_config(tmp_path, "model-smoke", "transformers")
    assert result_scope_note(tmp_path).startswith("Real-model smoke-test results")


def test_scripted_run_is_labeled_as_infrastructure_only(tmp_path):
    write_config(tmp_path, "smoke", "scripted")
    assert result_scope_note(tmp_path).startswith("Scripted-agent results")


def test_exact_mcnemar_known_cases():
    assert exact_mcnemar_pvalue(0, 0) == 1.0
    assert exact_mcnemar_pvalue(1, 0) == 1.0
    assert exact_mcnemar_pvalue(5, 0) == 0.0625
    assert exact_mcnemar_pvalue(3, 3) == 1.0
