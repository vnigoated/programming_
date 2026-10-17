from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from typing import Any

from .domains import build_environment
from .models import DecisionModel


IMPLEMENTED_POLICIES = {"serial", "naive", "occ", "txnguard"}


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
    if spec.policy in {"coagent", "mtpo_reproduction", "postgres_serializable"}:
        raise NotImplementedError(
            f"{spec.policy} is an explicit publication gate. Implement its real adapter before final runs."
        )
    if spec.policy not in IMPLEMENTED_POLICIES:
        raise ValueError(f"Unknown policy: {spec.policy}")
    if spec.fault_mode not in {"none", "duplicate_effect", "dependency_change"}:
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
    start = time.perf_counter()
    snapshots: dict[int, dict[str, Any]] = {}

    def observe(agent: int) -> dict[str, Any]:
        observation = environment.observe(agent)
        trace.append({"event": "read", "agent": agent, "observation": dict(observation)})
        return observation

    if spec.policy == "serial":
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

        if spec.policy in {"occ", "txnguard"} and environment.is_stale(observation):
            conflicts += 1
            trace.append(
                {"event": "conflict", "agent": agent, "resource_id": observation["resource_id"]}
            )
            observation = observe(agent)
            if not environment.can_apply(observation, decision.amount):
                rejected += 1
                continue

        if spec.policy == "txnguard" and not environment.can_apply(observation, decision.amount):
            rejected += 1
            trace.append(
                {
                    "event": "invariant_reject",
                    "agent": agent,
                    "resource_id": observation["resource_id"],
                }
            )
            continue

        idempotency_key = (
            f"{spec.run_id}:{agent}:{task['action']}:{observation['resource_id']}"
        )
        try:
            state = environment.commit(
                agent=agent,
                observation=observation,
                amount=decision.amount,
                stale_write=spec.policy == "naive",
                idempotency_key=idempotency_key,
                guarded=spec.policy == "txnguard",
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
        if spec.fault_mode == "duplicate_effect":
            delivered = environment.replay_effect(
                idempotency_key=idempotency_key,
                guarded=spec.policy == "txnguard",
            )
            trace.append(
                {
                    "event": "effect_replay",
                    "agent": agent,
                    "idempotency_key": idempotency_key,
                    "delivered": delivered,
                }
            )

    oracle = environment.score()
    duration = time.perf_counter() - start
    feasible = environment.successful_action_capacity()
    task_success = accepted >= feasible and not oracle.severe and invalid_actions == 0
    summary = {
        **spec.identity(),
        "run_id": spec.run_id,
        "completed": True,
        "accepted": accepted,
        "rejected": rejected,
        "conflicts": conflicts,
        "invalid_actions": invalid_actions,
        "unsafe_proposals": unsafe_proposals,
        "task_success": task_success,
        "invariant_violation": bool(oracle.invariant_violations),
        "side_effect_violation": bool(oracle.side_effect_violations),
        "lost_update": bool(oracle.lost_updates),
        "dependency_violation": bool(oracle.dependency_violations),
        "severe_violation": oracle.severe,
        "serializable": not oracle.severe,
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
        },
    }
    trace.append({"event": "oracle", "summary": summary})
    return summary, trace
