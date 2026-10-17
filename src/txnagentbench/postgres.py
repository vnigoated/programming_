from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable


SERIALIZATION_FAILURE = "40001"


class SimulatedWorkerCrash(RuntimeError):
    """Benchmark-only failure raised inside a transaction before commit."""


def _psycopg():
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError(
            "PostgreSQL support requires: pip install -e '.[postgres]'"
        ) from exc
    return psycopg


@dataclass(frozen=True)
class TransactionAttempt:
    attempt: int
    status: str
    sqlstate: str | None
    isolation_level: str | None
    duration_seconds: float


@dataclass(frozen=True)
class TransactionResult:
    agent: int
    resource_id: str
    accepted: bool
    observation: dict[str, int | str]
    final_state: dict[str, int | str]
    attempts: tuple[TransactionAttempt, ...]

    @property
    def serialization_failures(self) -> int:
        return sum(item.sqlstate == SERIALIZATION_FAILURE for item in self.attempts)


class PostgresSerializableStore:
    """Real PostgreSQL SSI store used by the serializable publication baseline."""

    def __init__(self, dsn: str, max_retries: int = 5, retry_backoff_seconds: float = 0.01):
        self.dsn = dsn
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds

    def connect(self):
        psycopg = _psycopg()
        connection = psycopg.connect(self.dsn, autocommit=True)
        connection.isolation_level = psycopg.IsolationLevel.SERIALIZABLE
        return connection

    def ensure_schema(self) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS txnagentbench_resources (
                    run_id text NOT NULL,
                    resource_id text NOT NULL,
                    capacity bigint NOT NULL CHECK (capacity >= 0),
                    allocated bigint NOT NULL CHECK (allocated >= 0 AND allocated <= capacity),
                    version bigint NOT NULL DEFAULT 0,
                    dependency_generation bigint NOT NULL DEFAULT 0,
                    PRIMARY KEY (run_id, resource_id)
                )
                """
            )
            connection.execute(
                """
                ALTER TABLE txnagentbench_resources
                ADD COLUMN IF NOT EXISTS dependency_generation bigint NOT NULL DEFAULT 0
                """
            )

    def server_metadata(self) -> dict[str, str]:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT current_setting('server_version'), current_setting('server_version_num')"
            ).fetchone()
        return {"server_version": str(row[0]), "server_version_num": str(row[1])}

    def reset_run(self, run_id: str, resources: Iterable[dict[str, Any]]) -> None:
        self.ensure_schema()
        with self.connect() as connection:
            with connection.transaction():
                connection.execute(
                    "DELETE FROM txnagentbench_resources WHERE run_id = %s", (run_id,)
                )
                for resource in resources:
                    connection.execute(
                        """
                        INSERT INTO txnagentbench_resources
                            (run_id, resource_id, capacity, allocated, version, dependency_generation)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (
                            run_id,
                            resource["resource_id"],
                            int(resource["capacity"]),
                            int(resource["allocated"]),
                            int(resource.get("version", 0)),
                            int(resource.get("dependency_generation", 0)),
                        ),
                    )

    def read_all(self, run_id: str) -> list[dict[str, int | str]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT resource_id, capacity, allocated, version
                FROM txnagentbench_resources WHERE run_id = %s ORDER BY resource_id
                """,
                (run_id,),
            ).fetchall()
        return [
            {
                "resource_id": row[0],
                "capacity": int(row[1]),
                "allocated": int(row[2]),
                "version": int(row[3]),
            }
            for row in rows
        ]

    def bump_dependencies(self, run_id: str) -> None:
        """Commit an external dependency change in its own serializable transaction."""
        with self.connect() as connection:
            with connection.transaction():
                connection.execute(
                    """
                    UPDATE txnagentbench_resources
                    SET dependency_generation = dependency_generation + 1,
                        version = version + 1
                    WHERE run_id = %s
                    """,
                    (run_id,),
                )

    def allocate(
        self,
        run_id: str,
        agent: int,
        resource_id: str,
        amount: int,
        after_first_read: Callable[[], None] | None = None,
        decide: Callable[[dict[str, int | str]], str] | None = None,
        crash_before_commit: bool = False,
    ) -> TransactionResult:
        psycopg = _psycopg()
        history: list[TransactionAttempt] = []
        for attempt in range(1, self.max_retries + 2):
            started = time.perf_counter()
            isolation: str | None = None
            observation: dict[str, int | str] | None = None
            try:
                with self.connect() as connection:
                    with connection.transaction():
                        isolation = str(
                            connection.execute("SHOW transaction_isolation").fetchone()[0]
                        )
                        row = connection.execute(
                            """
                            SELECT capacity, allocated, version, dependency_generation
                            FROM txnagentbench_resources
                            WHERE run_id = %s AND resource_id = %s
                            """,
                            (run_id, resource_id),
                        ).fetchone()
                        if row is None:
                            raise KeyError(f"Unknown resource {resource_id!r} for run {run_id!r}")
                        observation = {
                            "resource_id": resource_id,
                            "capacity": int(row[0]),
                            "allocated": int(row[1]),
                            "version": int(row[2]),
                            "dependency_generation": int(row[3]),
                        }
                        if attempt == 1 and after_first_read is not None:
                            after_first_read()
                        action = (
                            decide(dict(observation))
                            if decide is not None
                            else (
                                "apply"
                                if int(row[0]) - int(row[1]) >= amount
                                else "reject"
                            )
                        )
                        if action not in {"apply", "reject"}:
                            raise ValueError(f"Unsupported transaction decision: {action!r}")
                        accepted = action == "apply" and int(row[0]) - int(row[1]) >= amount
                        if accepted:
                            updated = connection.execute(
                                """
                                UPDATE txnagentbench_resources
                                SET allocated = allocated + %s, version = version + 1
                                WHERE run_id = %s AND resource_id = %s
                                RETURNING capacity, allocated, version, dependency_generation
                                """,
                                (amount, run_id, resource_id),
                            ).fetchone()
                            final_state = {
                                "resource_id": resource_id,
                                "capacity": int(updated[0]),
                                "allocated": int(updated[1]),
                                "version": int(updated[2]),
                                "dependency_generation": int(updated[3]),
                            }
                            if crash_before_commit:
                                raise SimulatedWorkerCrash(
                                    "worker crashed after SQL update and before transaction commit"
                                )
                        else:
                            final_state = dict(observation)
                history.append(
                    TransactionAttempt(
                        attempt, "committed", None, isolation, time.perf_counter() - started
                    )
                )
                return TransactionResult(
                    agent, resource_id, accepted, observation, final_state, tuple(history)
                )
            except SimulatedWorkerCrash:
                history.append(
                    TransactionAttempt(
                        attempt,
                        "worker_crash_rolled_back",
                        None,
                        isolation,
                        time.perf_counter() - started,
                    )
                )
                if observation is None:
                    raise AssertionError("pre-commit crash occurred before the database read")
                return TransactionResult(
                    agent,
                    resource_id,
                    False,
                    observation,
                    dict(observation),
                    tuple(history),
                )
            except psycopg.Error as exc:
                sqlstate = getattr(exc, "sqlstate", None)
                history.append(
                    TransactionAttempt(
                        attempt,
                        "serialization_retry" if sqlstate == SERIALIZATION_FAILURE else "failed",
                        sqlstate,
                        isolation,
                        time.perf_counter() - started,
                    )
                )
                if sqlstate != SERIALIZATION_FAILURE or attempt > self.max_retries:
                    raise
                time.sleep(self.retry_backoff_seconds * attempt)
        raise AssertionError("unreachable")
