from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from typing import Any

from .domains import build_environment
from .models import DecisionModel
from .faults import FAULT_MODES, REPLAY_FAULTS


IMPLEMENTED_POLICIES = {
    "serial",
    "serial_recovery",
    "naive",
    "recovery_only",
    "occ",
    "occ_recovery",
    "postgres_serializable",
    "mtpo_reproduction",
    "txnguard",
    "txnguard_no_recovery",
}
POLICY_UNSUPPORTED_FAULTS: dict[str, set[str]] = {}

SERIAL_POLICIES = {"serial", "serial_recovery"}
VERSION_VALIDATING_POLICIES = {
    "occ",
    "occ_recovery",
    "txnguard",
    "txnguard_no_recovery",
}
INVARIANT_GUARD_POLICIES = {"txnguard", "txnguard_no_recovery"}
RECOVERY_POLICIES = {
    "serial_recovery",
    "occ_recovery",
    "recovery_only",
    "txnguard",
}
STALE_WRITE_POLICIES = {"naive", "recovery_only"}


@dataclass(frozen=True)
class RunSpec:
    domain: str
    policy: str
    agent_count: int
    contention: int
    model_name: str
    schedule_seed: int
    scenario_seed: int = 0
    fault_mode: str = "none"
    model_revision: str | None = None
    prompt_version: str | None = None
    quantization: str | None = None

    def identity(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}

    @property
    def run_id(self) -> str:
        raw = json.dumps(self.identity(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()[:20]


def run_benchmark(spec: RunSpec, model: DecisionModel) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if spec.policy == "postgres_serializable":
        from .postgres_benchmark import run_postgres_benchmark

        return run_postgres_benchmark(spec, model)
    if spec.policy == "mtpo_reproduction":
        from .mtpo import run_mtpo_reproduction

        return run_mtpo_reproduction(spec, model)
    if spec.policy == "coagent":
        raise NotImplementedError(
            f"{spec.policy} is an explicit publication gate. Implement its real adapter before final runs."
        )
    if spec.policy not in IMPLEMENTED_POLICIES:
        raise ValueError(f"Unknown policy: {spec.policy}")
    if spec.fault_mode not in FAULT_MODES:
        raise ValueError(f"Unknown fault mode: {spec.fault_mode}")

    environment = build_environment(
        spec.domain, spec.agent_count, spec.contention, spec.scenario_seed
    )
    rng = random.Random(spec.schedule_seed)
    order = list(range(spec.agent_count))
    rng.shuffle(order)
    trace: list[dict[str, Any]] = []
    accepted = 0
    rejected = 0
    conflicts = 0
    invalid_actions = 0
    unsafe_proposals = 0
    input_tokens = 0
    output_tokens = 0
    model_latency = 0.0
    recovery_attempts = 0
    recovery_successes = 0
    recovery_violations: list[str] = []
    start = time.perf_counter()
    snapshots: dict[int, dict[str, Any]] = {}

    def observe(agent: int) -> dict[str, Any]:
        observation = environment.observe(agent)
        trace.append({"event": "read", "agent": agent, "observation": dict(observation)})
        return observation

    if spec.policy in SERIAL_POLICIES:
        execution = [(agent, None) for agent in order]
    else:
        for agent in order:
            snapshots[agent] = observe(agent)
        execution = [(agent, snapshots[agent]) for agent in order]

    trace.extend(environment.inject_fault(spec.fault_mode))

    for agent, observation in execution:
        if observation is None:
            observation = observe(agent)
        task = environment.task(agent)
        decision = model.decide(observation, task)
        input_tokens += decision.input_tokens
        output_tokens += decision.output_tokens
        model_latency += decision.latency_seconds
        trace.append(
            {
                "event": "decision",
                "agent": agent,
                "task": task,
                "action": decision.action,
                "amount": decision.amount,
                "raw_output": decision.raw_output,
            }
        )
        if decision.action == "invalid":
            invalid_actions += 1
            rejected += 1
            continue
        if decision.action == "reject":
            rejected += 1
            continue
        if decision.amount != int(task["amount"]) or decision.amount <= 0:
            invalid_actions += 1
            rejected += 1
            trace.append(
                {
                    "event": "invalid_amount",
                    "agent": agent,
                    "expected": task["amount"],
                    "received": decision.amount,
                }
            )
            continue

        proposal_safe = environment.can_apply(observation, decision.amount)
        if not proposal_safe:
            unsafe_proposals += 1
            trace.append(
                {
                    "event": "unsafe_proposal",
                    "agent": agent,
                    "resource_id": observation["resource_id"],
                    "available": observation["available"],
                    "requested": decision.amount,
                }
            )

        if spec.policy in VERSION_VALIDATING_POLICIES and environment.is_stale(observation):
            conflicts += 1
            trace.append(
                {"event": "conflict", "agent": agent, "resource_id": observation["resource_id"]}
            )
            observation = observe(agent)
            if not environment.can_apply(observation, decision.amount):
                rejected += 1
                continue

        if spec.policy in INVARIANT_GUARD_POLICIES and not environment.can_apply(
            observation, decision.amount
        ):
            rejected += 1
            trace.append(
                {
                    "event": "invariant_reject",
                    "agent": agent,
                    "resource_id": observation["resource_id"],
                }
            )
            continue

        if spec.fault_mode == "crash_before_commit":
            recovery_attempts += 1
            if spec.policy in RECOVERY_POLICIES:
                trace.append(
                    {
                        "event": "recovery_record",
                        "agent": agent,
                        "resource_id": observation["resource_id"],
                    }
                )
            trace.append(
                {
                    "event": "worker_crash",
                    "phase": "before_commit",
                    "agent": agent,
                    "resource_id": observation["resource_id"],
                }
            )
            if spec.policy not in RECOVERY_POLICIES:
                rejected += 1
                recovery_violations.append(f"unrecovered_precommit:{agent}")
                continue
            recovery_successes += 1
            trace.append({"event": "recovery_resume", "agent": agent})

        idempotency_key = (
            f"{spec.run_id}:{agent}:{task['action']}:{observation['resource_id']}"
        )
        try:
            state = environment.commit(
                agent=agent,
                observation=observation,
                amount=decision.amount,
                stale_write=spec.policy in STALE_WRITE_POLICIES,
                idempotency_key=idempotency_key,
                guarded=spec.policy in INVARIANT_GUARD_POLICIES,
            )
        except ValueError as exc:
            rejected += 1
            trace.append(
                {
                    "event": "invariant_reject",
                    "agent": agent,
                    "resource_id": observation["resource_id"],
                    "reason": str(exc),
                }
            )
            continue
        accepted += 1
        trace.append(
            {
                "event": "commit",
                "agent": agent,
                "resource_id": observation["resource_id"],
                "state": state,
            }
        )
        if spec.fault_mode in REPLAY_FAULTS:
            recovery_attempts += 1
            if spec.fault_mode == "crash_after_commit_before_ack":
                trace.append(
                    {"event": "worker_crash", "phase": "after_commit_before_ack", "agent": agent}
                )
            elif spec.fault_mode == "delayed_ack":
                trace.append({"event": "ack_timeout", "agent": agent})
            delivered = environment.replay_effect(
                idempotency_key=idempotency_key,
                guarded=spec.policy in RECOVERY_POLICIES,
            )
            trace.append(
                {
                    "event": "effect_replay",
                    "agent": agent,
                    "idempotency_key": idempotency_key,
                    "delivered": delivered,
                }
            )
            if delivered:
                recovery_violations.append(f"duplicate_effect:{agent}")
            else:
                recovery_successes += 1
        elif spec.fault_mode == "compensation_failure":
            recovery_attempts += 1
            compensated = environment.compensate(idempotency_key, fail=True)
            trace.append(
                {
                    "event": "compensation",
                    "agent": agent,
                    "attempt": 1,
                    "completed": compensated,
                }
            )
            if spec.policy in RECOVERY_POLICIES:
                compensated = environment.compensate(idempotency_key, fail=False)
                trace.append(
                    {
                        "event": "compensation",
                        "agent": agent,
                        "attempt": 2,
                        "completed": compensated,
                    }
                )
            if compensated:
                accepted -= 1
                rejected += 1
                recovery_successes += 1
            else:
                recovery_violations.append(f"compensation_failed:{agent}")

    oracle = environment.score()
    duration = time.perf_counter() - start
    feasible = (
        0
        if spec.fault_mode == "compensation_failure"
        else environment.successful_action_capacity()
    )
    severe = oracle.severe or bool(recovery_violations)
    task_success = accepted >= feasible and not severe and invalid_actions == 0
    summary = {
        **spec.identity(),
        "run_id": spec.run_id,
        "completed": True,
        "accepted": accepted,
        "rejected": rejected,
        "conflicts": conflicts,
        "invalid_actions": invalid_actions,
        "unsafe_proposals": unsafe_proposals,
        "recovery_attempts": recovery_attempts,
        "recovery_successes": recovery_successes,
        "recovery_rate": (
            recovery_successes / recovery_attempts if recovery_attempts else None
        ),
        "recovery_violation": bool(recovery_violations),
        "task_success": task_success,
        "invariant_violation": bool(oracle.invariant_violations),
        "side_effect_violation": bool(oracle.side_effect_violations),
        "lost_update": bool(oracle.lost_updates),
        "dependency_violation": bool(oracle.dependency_violations),
        "severe_violation": severe,
        "serializable": not severe,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model_latency_seconds": model_latency,
        "duration_seconds": duration,
        "final_state": environment.final_state(),
        "side_effect_count": environment.side_effect_count(),
        "oracle_details": {
            "invariant_violations": oracle.invariant_violations,
            "side_effect_violations": oracle.side_effect_violations,
            "lost_updates": oracle.lost_updates,
            "dependency_violations": oracle.dependency_violations,
            "recovery_violations": recovery_violations,
        },
    }
    trace.append({"event": "oracle", "summary": summary})
    return summary, trace
