import os
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from txnagentbench.postgres import PostgresSerializableStore
from txnagentbench.benchmark import RunSpec, run_benchmark
from txnagentbench.models import ScriptedDecisionModel


DSN = os.environ.get("TXNAGENTBENCH_TEST_POSTGRES_DSN")
pytestmark = pytest.mark.postgres


@pytest.fixture()
def store():
    if not DSN:
        pytest.skip("Set TXNAGENTBENCH_TEST_POSTGRES_DSN to run PostgreSQL integration tests")
    return PostgresSerializableStore(DSN, max_retries=5, retry_backoff_seconds=0)


def test_server_is_real_postgresql(store):
    metadata = store.server_metadata()
    assert int(metadata["server_version_num"]) >= 140000


def test_serializable_contention_retries_and_preserves_capacity(store):
    run_id = "integration-contention"
    store.reset_run(
        run_id,
        [{"resource_id": "shared", "capacity": 1, "allocated": 0, "version": 0}],
    )
    barrier = Barrier(2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(store.allocate, run_id, agent, "shared", 1, barrier.wait)
            for agent in range(2)
        ]
        results = [future.result(timeout=10) for future in futures]

    assert sum(result.accepted for result in results) == 1
    assert sum(result.serialization_failures for result in results) >= 1
    assert all(
        attempt.isolation_level == "serializable"
        for result in results
        for attempt in result.attempts
    )
    assert store.read_all(run_id) == [
        {"resource_id": "shared", "capacity": 1, "allocated": 1, "version": 1}
    ]


def test_zero_contention_commits_without_retry(store):
    run_id = "integration-private"
    store.reset_run(
        run_id,
        [
            {"resource_id": "private-0", "capacity": 1, "allocated": 0, "version": 0},
            {"resource_id": "private-1", "capacity": 1, "allocated": 0, "version": 0},
        ],
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda args: store.allocate(run_id, args[0], args[1], 1),
                [(0, "private-0"), (1, "private-1")],
            )
        )
    assert all(result.accepted for result in results)
    assert sum(result.serialization_failures for result in results) == 0


def test_precommit_worker_crash_rolls_back_sql_update(store):
    run_id = "integration-precommit-crash"
    store.reset_run(
        run_id,
        [{"resource_id": "shared", "capacity": 1, "allocated": 0, "version": 0}],
    )
    result = store.allocate(
        run_id, 0, "shared", 1, crash_before_commit=True
    )
    assert result.accepted is False
    assert result.attempts[-1].status == "worker_crash_rolled_back"
    assert store.read_all(run_id) == [
        {"resource_id": "shared", "capacity": 1, "allocated": 0, "version": 0}
    ]


def test_postgres_policy_runs_through_benchmark(monkeypatch, store):
    monkeypatch.setenv("TXNAGENTBENCH_POSTGRES_DSN", DSN)
    spec = RunSpec(
        "inventory", "postgres_serializable", 2, 100, "scripted-v1", 0
    )
    summary, trace = run_benchmark(spec, ScriptedDecisionModel())
    assert summary["task_success"] is True
    assert summary["severe_violation"] is False
    assert summary["accepted"] == 1
    assert summary["serialization_retries"] >= 1
    assert summary["database"]["isolation_level"] == "serializable"
    assert any(
        event.get("event") == "database_transaction" and event.get("sqlstate") == "40001"
        for event in trace
    )


def test_postgres_dependency_change_retries_to_current_generation(monkeypatch, store):
    monkeypatch.setenv("TXNAGENTBENCH_POSTGRES_DSN", DSN)
    spec = RunSpec(
        "cloud",
        "postgres_serializable",
        2,
        100,
        "scripted-v1",
        0,
        scenario_seed=1,
        fault_mode="dependency_change",
    )
    summary, trace = run_benchmark(spec, ScriptedDecisionModel())
    assert summary["task_success"] is True
    assert summary["dependency_violation"] is False
    assert summary["serialization_retries"] >= 1
    assert any(event.get("event") == "dependency_change" for event in trace)


@pytest.mark.parametrize(
    "fault_mode",
    [
        "crash_before_commit",
        "crash_after_commit_before_ack",
        "delayed_ack",
        "compensation_failure",
    ],
)
def test_postgres_reports_workflow_recovery_gap(monkeypatch, store, fault_mode):
    monkeypatch.setenv("TXNAGENTBENCH_POSTGRES_DSN", DSN)
    spec = RunSpec(
        "inventory",
        "postgres_serializable",
        2,
        0,
        "scripted-v1",
        0,
        scenario_seed=0,
        fault_mode=fault_mode,
    )
    summary, _ = run_benchmark(spec, ScriptedDecisionModel())
    assert summary["recovery_attempts"] > 0
    assert summary["recovery_rate"] == 0.0
    assert summary["recovery_violation"] is True
    assert summary["severe_violation"] is True
    if fault_mode == "crash_before_commit":
        assert summary["atomic_rollbacks"] == 2
        assert all(row["allocated"] == 0 for row in summary["database_final_state"])
