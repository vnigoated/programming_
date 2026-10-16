from txnagentbench.benchmark import RunSpec, run_benchmark
from txnagentbench.domains import build_environment
from txnagentbench.models import Decision, ScriptedDecisionModel
from txnagentbench.runner import expand_matrix


DOMAINS = ["inventory", "calendar", "support", "cloud"]


def result(domain: str, policy: str, contention: int = 100, scenario_seed: int = 0):
    spec = RunSpec(domain, policy, 2, contention, "scripted-v1", 0, scenario_seed)
    return run_benchmark(spec, ScriptedDecisionModel())[0]


def test_naive_exposes_domain_specific_failure_under_contention():
    for domain in DOMAINS:
        row = result(domain, "naive")
        assert row["severe_violation"] is True
        assert row["side_effect_violation"] is True
        assert row["lost_update"] is True
        assert row["oracle_details"]["side_effect_violations"]


def test_protected_policies_are_safe_in_every_domain():
    for domain in DOMAINS:
        for policy in ["serial", "occ", "txnguard"]:
            row = result(domain, policy)
            assert row["severe_violation"] is False
            assert row["task_success"] is True


def test_zero_contention_is_safe_in_every_domain():
    for domain in DOMAINS:
        row = result(domain, "naive", contention=0)
        assert row["severe_violation"] is False
        assert row["task_success"] is True


def test_domain_observations_and_tasks_are_not_aliases():
    observations = {}
    actions = set()
    for domain in DOMAINS:
        environment = build_environment(domain, 2, 100, scenario_seed=1)
        observation = environment.observe(0)
        task = environment.task(0)
        observations[domain] = set(observation)
        actions.add(task["action"])
    assert len(actions) == len(DOMAINS)
    assert "sku" in observations["inventory"]
    assert "booking_ids" in observations["calendar"]
    assert "eligible_cents" in observations["support"]
    assert "dependency_generation" in observations["cloud"]


def test_scenario_seed_changes_run_identity_and_support_amount():
    first = RunSpec("support", "serial", 2, 100, "scripted-v1", 0, 0)
    second = RunSpec("support", "serial", 2, 100, "scripted-v1", 0, 1)
    assert first.run_id != second.run_id
    assert build_environment("support", 2, 100, 0).task(0)["amount"] != build_environment(
        "support", 2, 100, 1
    ).task(0)["amount"]


class WrongAmountModel:
    def decide(self, observation, task):
        return Decision(action="apply", amount=int(task["amount"]) + 1, raw_output="wrong")

    def close(self):
        return None


def test_model_cannot_change_transaction_amount():
    spec = RunSpec("support", "txnguard", 2, 100, "wrong-amount", 0)
    row, trace = run_benchmark(spec, WrongAmountModel())
    assert row["invalid_actions"] == 2
    assert row["accepted"] == 0
    assert any(event["event"] == "invalid_amount" for event in trace)


def test_model_experiment_rejects_mutable_revision():
    config = {
        "stage": "pilot",
        "domains": ["inventory"],
        "policies": ["serial"],
        "agent_counts": [2],
        "contention_levels": [0],
        "models": [
            {
                "name": "mutable-model",
                "provider": "transformers",
                "model_id": "example/model",
                "revision": "main",
            }
        ],
        "schedule_seeds": [0],
    }
    try:
        expand_matrix(config)
    except ValueError as exc:
        assert "immutable revision" in str(exc)
    else:
        raise AssertionError("A mutable model revision was accepted")
