from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from threading import Barrier, Lock
from typing import Any

from .benchmark import RunSpec
from .domains import build_environment
from .faults import REPLAY_FAULTS
from .models import DecisionModel
from .postgres import PostgresSerializableStore, TransactionResult


def run_postgres_benchmark(
    spec: RunSpec, model: DecisionModel
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dsn = os.environ.get("TXNAGENTBENCH_POSTGRES_DSN")
    if not dsn:
        raise RuntimeError(
            "postgres_serializable requires TXNAGENTBENCH_POSTGRES_DSN and the postgres extra."
        )
    environment = build_environment(
        spec.domain, spec.agent_count, spec.contention, spec.scenario_seed
    )
    store = PostgresSerializableStore(dsn)
    initial = [environment.observe(agent) for agent in range(spec.agent_count)]
    resources = {observation["resource_id"]: observation for observation in initial}
    store.reset_run(spec.run_id, resources.values())
    server = store.server_metadata()

    trace: list[dict[str, Any]] = []
    model_lock = Lock()
    fault_events: list[dict[str, Any]] = []

    def inject_dependency_change() -> None:
        if spec.fault_mode == "dependency_change" and spec.domain == "cloud":
            fault_events.extend(environment.inject_fault(spec.fault_mode))
            store.bump_dependencies(spec.run_id)

    barrier = Barrier(spec.agent_count, action=inject_dependency_change)
    decision_records: dict[int, list[dict[str, Any]]] = {
        agent: [] for agent in range(spec.agent_count)
    }
    start = time.perf_counter()

    def worker(agent: int) -> TransactionResult:
        task = environment.task(agent)
        resource_id = initial[agent]["resource_id"]

        def decide(database_observation: dict[str, int | str]) -> str:
            observation = {
                **database_observation,
                "available": int(database_observation["capacity"])
                - int(database_observation["allocated"]),
                "scenario_family": initial[agent].get("scenario_family"),
            }
            with model_lock:
                decision = model.decide(observation, task)
            record = {
                "event": "decision",
                "agent": agent,
                "task": task,
                "observation": observation,
                "action": decision.action,
                "amount": decision.amount,
                "raw_output": decision.raw_output,
                "input_tokens": decision.input_tokens,
                "output_tokens": decision.output_tokens,
                "model_latency_seconds": decision.latency_seconds,
            }
            decision_records[agent].append(record)
            if decision.action == "apply" and decision.amount == int(task["amount"]):
                return "apply"
            return "reject"

        return store.allocate(
            spec.run_id,
            agent,
            resource_id,
            int(task["amount"]),
            after_first_read=barrier.wait,
            decide=decide,
            crash_before_commit=spec.fault_mode == "crash_before_commit",
        )

    with ThreadPoolExecutor(max_workers=spec.agent_count) as pool:
        futures = [pool.submit(worker, agent) for agent in range(spec.agent_count)]
        results = [future.result() for future in futures]

    trace.extend(fault_events)

    accepted = 0
    rejected = 0
    invalid_actions = 0
    unsafe_proposals = 0
    conflicts = 0
    input_tokens = 0
    output_tokens = 0
    model_latency = 0.0
    recovery_attempts = 0
    recovery_successes = 0
    recovery_violations: list[str] = []
    atomic_rollbacks = 0

    for result in sorted(results, key=lambda item: item.agent):
        records = decision_records[result.agent]
        for record in records:
            trace.append(
                {
                    "event": "read",
                    "agent": result.agent,
                    "observation": record["observation"],
                }
            )
            trace.append(record)
            input_tokens += int(record["input_tokens"])
            output_tokens += int(record["output_tokens"])
            model_latency += float(record["model_latency_seconds"])
            if record["action"] == "invalid" or record["amount"] != int(record["task"]["amount"]):
                invalid_actions += 1
            if record["action"] == "apply" and int(record["observation"]["available"]) < int(
                record["task"]["amount"]
            ):
                unsafe_proposals += 1
        for attempt in result.attempts:
            trace.append(
                {
                    "event": "database_transaction",
                    "agent": result.agent,
                    **asdict(attempt),
                }
            )
        conflicts += result.serialization_failures
        if any(attempt.status == "worker_crash_rolled_back" for attempt in result.attempts):
            recovery_attempts += 1
            atomic_rollbacks += 1
            recovery_violations.append(f"unrecovered_precommit:{result.agent}")
            trace.append(
                {
                    "event": "worker_crash",
                    "phase": "before_commit",
                    "agent": result.agent,
                    "database_rolled_back": True,
                }
            )
        if not result.accepted:
            rejected += 1
            continue
        accepted += 1
        observation = environment.observe(result.agent)
        task = environment.task(result.agent)
        idempotency_key = (
            f"{spec.run_id}:{result.agent}:{task['action']}:{observation['resource_id']}"
        )
        state = environment.commit(
            agent=result.agent,
            observation=observation,
            amount=int(task["amount"]),
            stale_write=False,
            idempotency_key=idempotency_key,
            guarded=True,
        )
        trace.append(
            {
                "event": "commit",
                "agent": result.agent,
                "resource_id": observation["resource_id"],
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
                        "agent": result.agent,
                    }
                )
            elif spec.fault_mode == "delayed_ack":
                trace.append({"event": "ack_timeout", "agent": result.agent})
            delivered = environment.replay_effect(idempotency_key=idempotency_key, guarded=False)
            trace.append(
                {
                    "event": "effect_replay",
                    "agent": result.agent,
                    "idempotency_key": idempotency_key,
                    "delivered": delivered,
                }
            )
            if delivered:
                recovery_violations.append(f"duplicate_effect:{result.agent}")
            else:
                recovery_successes += 1
        elif spec.fault_mode == "compensation_failure":
            recovery_attempts += 1
            compensated = environment.compensate(idempotency_key, fail=True)
            trace.append(
                {
                    "event": "compensation",
                    "agent": result.agent,
                    "attempt": 1,
                    "completed": compensated,
                }
            )
            recovery_violations.append(f"compensation_failed:{result.agent}")

    oracle = environment.score()
    duration = time.perf_counter() - start
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
        "conflicts": conflicts,
        "serialization_retries": conflicts,
        "invalid_actions": invalid_actions,
        "unsafe_proposals": unsafe_proposals,
        "recovery_attempts": recovery_attempts,
        "recovery_successes": recovery_successes,
        "recovery_rate": (
            recovery_successes / recovery_attempts if recovery_attempts else None
        ),
        "recovery_violation": bool(recovery_violations),
        "atomic_rollbacks": atomic_rollbacks,
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
        "duration_seconds": duration,
        "final_state": environment.final_state(),
        "database_final_state": store.read_all(spec.run_id),
        "database": {**server, "isolation_level": "serializable"},
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
