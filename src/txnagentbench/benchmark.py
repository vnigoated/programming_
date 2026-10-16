from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from typing import Any

from .models import DecisionModel


DOMAINS = {
    "inventory": {"resource": "stock_unit", "effect": "shipment"},
    "calendar": {"resource": "meeting_slot", "effect": "invitation"},
    "support": {"resource": "refund_authorization", "effect": "refund"},
    "cloud": {"resource": "service_quota", "effect": "deployment"},
}


@dataclass(frozen=True)
class RunSpec:
    domain: str
    policy: str
    agent_count: int
    contention: int
    model_name: str
    schedule_seed: int

    @property
    def run_id(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()[:20]


def make_resources(spec: RunSpec) -> dict[str, dict[str, int]]:
    shared = 0 if spec.contention == 0 else max(2, round(spec.agent_count * spec.contention / 100))
    resources: dict[str, dict[str, int]] = {}
    for agent in range(spec.agent_count):
        key = "shared" if agent < shared else f"private-{agent}"
        resources.setdefault(key, {"capacity": 1, "allocated": 0, "version": 0})
    return resources


def resource_for_agent(spec: RunSpec, agent: int) -> str:
    shared = 0 if spec.contention == 0 else max(2, round(spec.agent_count * spec.contention / 100))
    return "shared" if agent < shared else f"private-{agent}"


def run_benchmark(spec: RunSpec, model: DecisionModel) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if spec.domain not in DOMAINS:
        raise ValueError(f"Unknown domain: {spec.domain}")
    if spec.policy == "coagent":
        raise NotImplementedError(
            "CoAgent is an explicit publication gate. Implement a faithful adapter before final runs."
        )
    if spec.policy not in {"serial", "naive", "occ", "txnguard"}:
        raise ValueError(f"Unknown policy: {spec.policy}")

    rng = random.Random(spec.schedule_seed)
    resources = make_resources(spec)
    order = list(range(spec.agent_count))
    rng.shuffle(order)
    trace: list[dict[str, Any]] = []
    side_effects: list[dict[str, Any]] = []
    accepted = 0
    rejected = 0
    conflicts = 0
    invalid_actions = 0
    input_tokens = 0
    output_tokens = 0
    model_latency = 0.0
    start = time.perf_counter()

    snapshots: dict[int, dict[str, Any]] = {}

    def observe(agent: int) -> dict[str, Any]:
        key = resource_for_agent(spec, agent)
        state = resources[key]
        obs = {"resource_id": key, **state}
        trace.append({"event": "read", "agent": agent, "observation": dict(obs)})
        return obs

    if spec.policy == "serial":
        # Observe immediately before each commit so later agents see prior commits.
        execution = [(agent, None) for agent in order]
    else:
        for agent in order:
            snapshots[agent] = observe(agent)
        execution = [(agent, snapshots[agent]) for agent in order]

    for agent, observation in execution:
        if observation is None:
            observation = observe(agent)
        task = {"action": "consume", "amount": 1, "domain": spec.domain}
        decision = model.decide(observation, task)
        input_tokens += decision.input_tokens
        output_tokens += decision.output_tokens
        model_latency += decision.latency_seconds
        trace.append(
            {
                "event": "decision",
                "agent": agent,
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

        key = observation["resource_id"]
        state = resources[key]
        stale = state["version"] != observation["version"]
        idempotency_key = f"{spec.run_id}:{agent}:{DOMAINS[spec.domain]['effect']}"

        if spec.policy in {"occ", "txnguard"} and stale:
            conflicts += 1
            trace.append({"event": "conflict", "agent": agent, "resource_id": key})
            fresh = observe(agent)
            if fresh["allocated"] >= fresh["capacity"]:
                rejected += 1
                continue
            observation = fresh

        candidate = state["allocated"] + decision.amount
        if spec.policy == "txnguard" and candidate > state["capacity"]:
            rejected += 1
            trace.append({"event": "invariant_reject", "agent": agent, "resource_id": key})
            continue

        if spec.policy == "naive":
            # Write from the stale snapshot to reproduce a lost-update history.
            state["allocated"] = observation["allocated"] + decision.amount
        else:
            state["allocated"] = candidate
        state["version"] += 1
        accepted += 1
        effect = {
            "agent": agent,
            "resource_id": key,
            "effect": DOMAINS[spec.domain]["effect"],
            "idempotency_key": idempotency_key,
        }
        if spec.policy != "txnguard" or not any(
            item["idempotency_key"] == idempotency_key for item in side_effects
        ):
            side_effects.append(effect)
        trace.append({"event": "commit", "agent": agent, "resource_id": key, "state": dict(state)})

    violations = [key for key, state in resources.items() if state["allocated"] > state["capacity"]]
    effects_by_resource: dict[str, int] = {}
    for effect in side_effects:
        effects_by_resource[effect["resource_id"]] = effects_by_resource.get(effect["resource_id"], 0) + 1
    side_effect_violations = [
        key for key, count in effects_by_resource.items() if count > resources[key]["capacity"]
    ]
    lost_updates = [
        key
        for key, count in effects_by_resource.items()
        if count != resources[key]["allocated"]
    ]
    duration = time.perf_counter() - start
    severe = bool(violations or side_effect_violations or lost_updates)
    feasible = sum(state["capacity"] for state in resources.values())
    task_success = accepted >= feasible and not severe and invalid_actions == 0
    summary = {
        **asdict(spec),
        "run_id": spec.run_id,
        "completed": True,
        "accepted": accepted,
        "rejected": rejected,
        "conflicts": conflicts,
        "invalid_actions": invalid_actions,
        "task_success": task_success,
        "invariant_violation": bool(violations),
        "side_effect_violation": bool(side_effect_violations),
        "lost_update": bool(lost_updates),
        "severe_violation": severe,
        "serializable": not severe,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model_latency_seconds": model_latency,
        "duration_seconds": duration,
        "final_state": resources,
        "side_effect_count": len(side_effects),
    }
    trace.append({"event": "oracle", "summary": summary})
    return summary, trace

