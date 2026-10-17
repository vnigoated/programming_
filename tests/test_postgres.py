import os
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from txnagentbench.postgres import PostgresSerializableStore


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
