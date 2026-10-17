from __future__ import annotations

import random
import time
from typing import Any

from .benchmark import RunSpec
from .domains import build_environment
from .faults import REPLAY_FAULTS
from .models import Decision, DecisionModel


def _premise(observation: dict[str, Any]) -> tuple[Any, ...]:
    return (
        observation.get("resource_id"),
        observation.get("allocated"),
        observation.get("capacity"),
        observation.get("version"),
        observation.get("dependency_generation"),
    )


def run_mtpo_reproduction(
    spec: RunSpec, model: DecisionModel
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Independent benchmark-level reproduction of the paper-specified MTPO protocol."""
    environment = build_environment(
        spec.domain, spec.agent_count, spec.contention, spec.scenario_seed
    )
    trace: list[dict[str, Any]] = []
    start = time.perf_counter()
    observations: dict[int, dict[str, Any]] = {}
    initial_decisions: dict[int, Decision] = {}
    input_tokens = output_tokens = 0
    model_latency = 0.0
    invalid_actions = unsafe_proposals = 0
    recovery_attempts = recovery_successes = 0
    recovery_violations: list[str] = []

    physical_order = list(range(spec.agent_count))
    random.Random(spec.schedule_seed).shuffle(physical_order)
    ranks = {agent: agent for agent in range(spec.agent_count)}
    trace.append(
        {
            "event": "mtpo_launch",
            "serialization_ranks": ranks,
            "physical_order": physical_order,
            "implementation": "independent_reproduction",
        }
    )

    for agent in physical_order:
        observation = environment.observe(agent)
        observations[agent] = observation
        trace.append({"event": "read", "agent": agent, "observation": observation})
        task = environment.task(agent)
        decision = model.decide(observation, task)
        initial_decisions[agent] = decision
        input_tokens += decision.input_tokens
        output_tokens += decision.output_tokens
        model_latency += decision.latency_seconds
        trace.append(
            {
                "event": "decision",
                "phase": "speculative",
                "agent": agent,
                "task": task,
                "action": decision.action,
                "amount": decision.amount,
                "raw_output": decision.raw_output,
            }
        )
        if decision.action == "invalid" or decision.amount != int(task["amount"]):
            invalid_actions += 1
        elif decision.action == "apply" and not environment.can_apply(
            observation, decision.amount
        ):
            unsafe_proposals += 1
        trace.append(
            {
                "event": "trajectory_append",
                "agent": agent,
                "rank": ranks[agent],
                "resource_id": observation["resource_id"],
                "action": decision.action,
            }
        )

    trace.extend(environment.inject_fault(spec.fault_mode))
    accepted = rejected = notifications = repairs = inverse_operations = 0

    # The benchmark environments expose resettable synthetic state. Reconciliation replays the
    # registered operations in fixed rank order, which is the explicit benchmark-level equivalent
    # of undoing and reordering the speculative trajectory.
    for agent in sorted(ranks, key=ranks.get):
        task = environment.task(agent)
        current = environment.observe(agent)
        decision = initial_decisions[agent]
        if spec.fault_mode == "crash_before_commit":
            recovery_attempts += 1
            recovery_successes += 1
            trace.extend(
                [
                    {
                        "event": "worker_crash",
                        "phase": "before_commit",
                        "agent": agent,
                    },
                    {
                        "event": "trajectory_recovery",
                        "agent": agent,
                        "rank": ranks[agent],
                    },
                ]
            )
        changed = _premise(current) != _premise(observations[agent])
        if changed:
            notifications += 1
            inverse_operations += int(decision.action == "apply")
            trace.append(
                {
                    "event": "mtpo_notification",
                    "agent": agent,
                    "rank": ranks[agent],
                    "old_premise": observations[agent],
                    "current_premise": current,
                }
            )
            repaired = model.decide(current, task)
            input_tokens += repaired.input_tokens
            output_tokens += repaired.output_tokens
            model_latency += repaired.latency_seconds
            repairs += 1
            decision = repaired
            trace.append(
                {
                    "event": "decision",
                    "phase": "semantic_repair",
                    "agent": agent,
                    "task": task,
                    "action": decision.action,
                    "amount": decision.amount,
                    "raw_output": decision.raw_output,
                }
            )

        if decision.action == "invalid" or decision.amount != int(task["amount"]):
            rejected += 1
            continue
        if decision.action != "apply" or not environment.can_apply(current, decision.amount):
            rejected += 1
            continue
        idempotency_key = (
            f"{spec.run_id}:{agent}:{task['action']}:{current['resource_id']}"
        )
        state = environment.commit(
            agent=agent,
            observation=current,
            amount=decision.amount,
            stale_write=False,
            idempotency_key=idempotency_key,
            guarded=True,
        )
        accepted += 1
        trace.append(
            {
                "event": "commit",
                "agent": agent,
                "rank": ranks[agent],
                "resource_id": current["resource_id"],
                "state": state,
            }
        )
        if spec.fault_mode in REPLAY_FAULTS:
            recovery_attempts += 1
            if spec.fault_mode == "crash_after_commit_before_ack":
                trace.append(
                    {
                        "event": "worker_crash",
                        "phase": "after_commit_before_ack",
                        "agent": agent,
                    }
                )
            elif spec.fault_mode == "delayed_ack":
                trace.append({"event": "ack_timeout", "agent": agent})
            delivered = environment.replay_effect(idempotency_key, guarded=True)
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
                inverse_operations += 1
                recovery_successes += 1
            else:
                recovery_violations.append(f"compensation_failed:{agent}")

    trace.append(
        {
            "event": "trajectory_reconcile",
            "notifications": notifications,
            "semantic_repairs": repairs,
            "inverse_operations": inverse_operations,
        }
    )
    oracle = environment.score()
    feasible = (
        0
        if spec.fault_mode == "compensation_failure"
        else environment.successful_action_capacity()
    )
    severe = oracle.severe or bool(recovery_violations)
    summary = {
        **spec.identity(),
        "run_id": spec.run_id,
        "completed": True,
        "accepted": accepted,
        "rejected": rejected,
        "conflicts": notifications,
        "mtpo_notifications": notifications,
        "semantic_repairs": repairs,
        "inverse_operations": inverse_operations,
        "invalid_actions": invalid_actions,
        "unsafe_proposals": unsafe_proposals,
        "recovery_attempts": recovery_attempts,
        "recovery_successes": recovery_successes,
        "recovery_rate": (
            recovery_successes / recovery_attempts if recovery_attempts else None
        ),
        "recovery_violation": bool(recovery_violations),
        "task_success": accepted >= feasible and not severe and invalid_actions == 0,
        "invariant_violation": bool(oracle.invariant_violations),
        "side_effect_violation": bool(oracle.side_effect_violations),
        "lost_update": bool(oracle.lost_updates),
        "dependency_violation": bool(oracle.dependency_violations),
        "severe_violation": severe,
        "serializable": not severe,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model_latency_seconds": model_latency,
        "duration_seconds": time.perf_counter() - start,
        "final_state": environment.final_state(),
        "side_effect_count": environment.side_effect_count(),
        "oracle_details": {
            "invariant_violations": oracle.invariant_violations,
            "side_effect_violations": oracle.side_effect_violations,
            "lost_updates": oracle.lost_updates,
            "dependency_violations": oracle.dependency_violations,
            "recovery_violations": recovery_violations,
        },
        "baseline_provenance": {
            "name": "mtpo_reproduction",
            "official_author_code": False,
            "protocol_source": "arXiv:2606.15376",
        },
    }
    trace.append({"event": "oracle", "summary": summary})
    return summary, trace
