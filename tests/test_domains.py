from txnagentbench.benchmark import RunSpec, run_benchmark
from txnagentbench.domains import build_environment
from txnagentbench.models import Decision, ScriptedDecisionModel
from txnagentbench.runner import expand_matrix


DOMAINS = ["inventory", "calendar", "support", "cloud"]


def result(domain: str, policy: str, contention: int = 100, scenario_seed: int = 0):
    spec = RunSpec(domain, policy, 2, contention, "scripted-v1", 0, scenario_seed)
    return run_benchmark(spec, ScriptedDecisionModel())[0]


def fault_result(domain: str, policy: str):
    spec = RunSpec(domain, policy, 2, 0, "scripted-v1", 0, 0, "duplicate_effect")
    return run_benchmark(spec, ScriptedDecisionModel())


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


def test_scenario_seed_changes_run_identity_and_support_capacity():
    first = RunSpec("support", "serial", 2, 100, "scripted-v1", 0, 0)
    second = RunSpec("support", "serial", 2, 100, "scripted-v1", 0, 1)
    assert first.run_id != second.run_id
    assert build_environment("support", 2, 100, 0).observe(0)["capacity"] != build_environment(
        "support", 2, 100, 1
    ).observe(0)["capacity"]


class WrongAmountModel:
    def decide(self, observation, task):
        return Decision(action="apply", amount=int(task["amount"]) + 1, raw_output="wrong")

    def close(self):
        return None


class AlwaysApplyModel:
    def decide(self, observation, task):
        return Decision(action="apply", amount=int(task["amount"]), raw_output="apply")

    def close(self):
        return None


def test_model_cannot_change_transaction_amount():
    spec = RunSpec("support", "txnguard", 2, 100, "wrong-amount", 0)
    row, trace = run_benchmark(spec, WrongAmountModel())
    assert row["invalid_actions"] == 2
    assert row["accepted"] == 0
    assert any(event["event"] == "invalid_amount" for event in trace)


def test_unsafe_serial_proposal_is_measured_and_txnguard_blocks_stale_commit():
    serial_spec = RunSpec("inventory", "serial", 2, 100, "always-apply", 0)
    serial, serial_trace = run_benchmark(serial_spec, AlwaysApplyModel())
    assert serial["unsafe_proposals"] == 1
    assert serial["severe_violation"] is True
    assert any(event["event"] == "unsafe_proposal" for event in serial_trace)

    guard_spec = RunSpec("inventory", "txnguard", 2, 100, "always-apply", 0)
    guarded, _ = run_benchmark(guard_spec, AlwaysApplyModel())
    assert guarded["unsafe_proposals"] == 0
    assert guarded["conflicts"] == 1
    assert guarded["severe_violation"] is False
    assert guarded["task_success"] is True


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


def test_occ_does_not_deduplicate_external_effect_replay():
    for domain in DOMAINS:
        row, trace = fault_result(domain, "occ")
        assert row["severe_violation"] is True
        assert row["side_effect_violation"] is True
        assert any(
            event["event"] == "effect_replay" and event["delivered"] is True
            for event in trace
        )


def test_txnguard_deduplicates_external_effect_replay():
    for domain in DOMAINS:
        row, trace = fault_result(domain, "txnguard")
        assert row["severe_violation"] is False
        assert row["task_success"] is True
        assert all(
            event["delivered"] is False
            for event in trace
            if event["event"] == "effect_replay"
        )


def test_fault_mode_changes_run_identity():
    baseline = RunSpec("inventory", "occ", 2, 0, "scripted-v1", 0)
    replay = RunSpec("inventory", "occ", 2, 0, "scripted-v1", 0, 0, "duplicate_effect")
    assert baseline.run_id != replay.run_id


def test_model_revision_and_prompt_change_run_identity():
    first = RunSpec(
        "inventory",
        "serial",
        2,
        100,
        "model-a",
        0,
        model_revision="a" * 40,
        prompt_version="allocation-v2",
        quantization="bf16",
    )
    second = RunSpec(
        "inventory",
        "serial",
        2,
        100,
        "model-a",
        0,
        model_revision="b" * 40,
        prompt_version="allocation-v2",
        quantization="bf16",
    )
    assert first.run_id != second.run_id


def test_oracles_reject_constructed_invalid_histories():
    inventory = build_environment("inventory", 2, 100)
    inventory.items["sku-shared"]["reserved"] = 2
    assert inventory.score().invariant_violations == ["sku-shared"]

    calendar = build_environment("calendar", 2, 100)
    slot = next(iter(calendar.slots))
    calendar.slots[slot]["booking_ids"] = ["meeting-a", "meeting-b"]
    assert calendar.score().invariant_violations == [slot]

    support = build_environment("support", 2, 100)
    case_id = next(iter(support.cases))
    support.cases[case_id]["refunded_cents"] = support.cases[case_id]["eligible_cents"] + 1
    assert support.score().invariant_violations == [case_id]

    cloud = build_environment("cloud", 2, 100)
    pool_id = next(iter(cloud.pools))
    cloud.pools[pool_id]["used_units"] = cloud.pools[pool_id]["quota_units"] + 1
    cloud.deployments.append(
        {
            "effect": "deployment",
            "pool_id": pool_id,
            "deployment_id": "bad-generation",
            "amount_units": 1,
            "dependency_generation": cloud.pools[pool_id]["dependency_generation"] - 1,
            "idempotency_key": "corrupt",
            "baseline": False,
        }
    )
    oracle = cloud.score()
    assert oracle.invariant_violations == [pool_id]
    assert oracle.dependency_violations == ["bad-generation"]


def test_fresh_domain_states_pass_all_oracles():
    for domain in DOMAINS:
        assert build_environment(domain, 2, 100).score().severe is False


def test_scenario_families_include_valid_rejections_and_multi_fit_capacity():
    for domain in ["inventory", "support", "cloud"]:
        two_fit = build_environment(domain, 2, 100, scenario_seed=1)
        assert two_fit.successful_action_capacity() == 2
        exhausted = build_environment(domain, 2, 100, scenario_seed=3)
        assert exhausted.successful_action_capacity() == 0
        assert exhausted.observe(0)["available"] == 0
    prebooked = build_environment("calendar", 2, 100, scenario_seed=3)
    assert prebooked.successful_action_capacity() == 0
    assert prebooked.observe(0)["available"] == 0


def test_cloud_dependency_change_is_detected_and_versioned_policies_recover():
    naive_spec = RunSpec(
        "cloud", "naive", 2, 100, "scripted-v1", 0, 0, "dependency_change"
    )
    naive, naive_trace = run_benchmark(naive_spec, ScriptedDecisionModel())
    assert naive["dependency_violation"] is True
    assert any(event["event"] == "dependency_change" for event in naive_trace)

    for scenario_seed in range(4):
        for policy in ["serial", "occ", "txnguard"]:
            spec = RunSpec(
                "cloud",
                policy,
                2,
                100,
                "scripted-v1",
                0,
                scenario_seed,
                "dependency_change",
            )
            row, _ = run_benchmark(spec, ScriptedDecisionModel())
            assert row["dependency_violation"] is False
            assert row["task_success"] is True
