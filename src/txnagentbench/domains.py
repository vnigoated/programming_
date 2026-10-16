from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


def shared_agent_count(agent_count: int, contention: int) -> int:
    if contention == 0:
        return 0
    return min(agent_count, max(2, round(agent_count * contention / 100)))


def assigned_key(prefix: str, agent: int, agent_count: int, contention: int) -> str:
    if agent < shared_agent_count(agent_count, contention):
        return f"{prefix}-shared"
    return f"{prefix}-{agent}"


@dataclass(frozen=True)
class OracleResult:
    invariant_violations: list[str]
    side_effect_violations: list[str]
    lost_updates: list[str]
    dependency_violations: list[str]

    @property
    def severe(self) -> bool:
        return bool(
            self.invariant_violations
            or self.side_effect_violations
            or self.lost_updates
            or self.dependency_violations
        )


class DomainEnvironment(ABC):
    name: str

    def __init__(self, agent_count: int, contention: int, scenario_seed: int):
        self.agent_count = agent_count
        self.contention = contention
        self.scenario_seed = scenario_seed

    @abstractmethod
    def observe(self, agent: int) -> dict[str, Any]: ...

    @abstractmethod
    def task(self, agent: int) -> dict[str, Any]: ...

    @abstractmethod
    def is_stale(self, observation: dict[str, Any]) -> bool: ...

    @abstractmethod
    def can_apply(self, observation: dict[str, Any], amount: int) -> bool: ...

    @abstractmethod
    def commit(
        self,
        agent: int,
        observation: dict[str, Any],
        amount: int,
        stale_write: bool,
        idempotency_key: str,
        guarded: bool,
    ) -> dict[str, Any]: ...

    @abstractmethod
    def score(self) -> OracleResult: ...

    @abstractmethod
    def final_state(self) -> dict[str, Any]: ...

    @abstractmethod
    def successful_action_capacity(self) -> int: ...

    @abstractmethod
    def side_effect_count(self) -> int: ...


class InventoryEnvironment(DomainEnvironment):
    name = "inventory"

    def __init__(self, agent_count: int, contention: int, scenario_seed: int):
        super().__init__(agent_count, contention, scenario_seed)
        self.items: dict[str, dict[str, int]] = {}
        self.shipments: list[dict[str, Any]] = []
        for agent in range(agent_count):
            sku = assigned_key("sku", agent, agent_count, contention)
            self.items.setdefault(sku, {"stock": 1, "reserved": 0, "version": 0})

    def _key(self, agent: int) -> str:
        return assigned_key("sku", agent, self.agent_count, self.contention)

    def observe(self, agent: int) -> dict[str, Any]:
        sku = self._key(agent)
        state = self.items[sku]
        return {
            "resource_id": sku,
            "sku": sku,
            "stock": state["stock"],
            "reserved": state["reserved"],
            "capacity": state["stock"],
            "allocated": state["reserved"],
            "version": state["version"],
        }

    def task(self, agent: int) -> dict[str, Any]:
        return {
            "action": "reserve_inventory",
            "amount": 1,
            "domain": self.name,
            "order_id": f"order-{agent}",
        }

    def is_stale(self, observation: dict[str, Any]) -> bool:
        return self.items[observation["sku"]]["version"] != observation["version"]

    def can_apply(self, observation: dict[str, Any], amount: int) -> bool:
        return observation["reserved"] + amount <= observation["stock"]

    def commit(self, agent, observation, amount, stale_write, idempotency_key, guarded):
        state = self.items[observation["sku"]]
        candidate = observation["reserved"] + amount if stale_write else state["reserved"] + amount
        if guarded and candidate > state["stock"]:
            raise ValueError("inventory_stock_invariant")
        state["reserved"] = candidate
        state["version"] += 1
        effect = {
            "effect": "shipment",
            "sku": observation["sku"],
            "order_id": f"order-{agent}",
            "idempotency_key": idempotency_key,
        }
        if not guarded or not any(x["idempotency_key"] == idempotency_key for x in self.shipments):
            self.shipments.append(effect)
        return dict(state)

    def score(self) -> OracleResult:
        invariant = [key for key, state in self.items.items() if state["reserved"] > state["stock"]]
        effect_counts = {key: 0 for key in self.items}
        for effect in self.shipments:
            effect_counts[effect["sku"]] += 1
        side = [key for key, count in effect_counts.items() if count > self.items[key]["stock"]]
        lost = [key for key, count in effect_counts.items() if count != self.items[key]["reserved"]]
        return OracleResult(invariant, side, lost, [])

    def final_state(self) -> dict[str, Any]:
        return {"items": self.items, "shipments": self.shipments}

    def successful_action_capacity(self) -> int:
        return sum(state["stock"] for state in self.items.values())

    def side_effect_count(self) -> int:
        return len(self.shipments)


class CalendarEnvironment(DomainEnvironment):
    name = "calendar"

    def __init__(self, agent_count: int, contention: int, scenario_seed: int):
        super().__init__(agent_count, contention, scenario_seed)
        self.slots: dict[str, dict[str, Any]] = {}
        self.invitations: list[dict[str, Any]] = []
        hour = 9 + scenario_seed % 8
        for agent in range(agent_count):
            slot = assigned_key(f"room-{hour:02d}00", agent, agent_count, contention)
            self.slots.setdefault(slot, {"booking_ids": [], "version": 0})

    def _key(self, agent: int) -> str:
        hour = 9 + self.scenario_seed % 8
        return assigned_key(f"room-{hour:02d}00", agent, self.agent_count, self.contention)

    def observe(self, agent: int) -> dict[str, Any]:
        slot = self._key(agent)
        state = self.slots[slot]
        return {
            "resource_id": slot,
            "slot_id": slot,
            "booking_ids": list(state["booking_ids"]),
            "capacity": 1,
            "allocated": len(state["booking_ids"]),
            "version": state["version"],
        }

    def task(self, agent: int) -> dict[str, Any]:
        return {
            "action": "book_meeting",
            "amount": 1,
            "domain": self.name,
            "meeting_id": f"meeting-{agent}",
        }

    def is_stale(self, observation: dict[str, Any]) -> bool:
        return self.slots[observation["slot_id"]]["version"] != observation["version"]

    def can_apply(self, observation: dict[str, Any], amount: int) -> bool:
        return len(observation["booking_ids"]) + amount <= 1

    def commit(self, agent, observation, amount, stale_write, idempotency_key, guarded):
        state = self.slots[observation["slot_id"]]
        current = list(observation["booking_ids"]) if stale_write else list(state["booking_ids"])
        if guarded and len(current) + amount > 1:
            raise ValueError("calendar_exclusivity_invariant")
        current.append(f"meeting-{agent}")
        state["booking_ids"] = current
        state["version"] += 1
        effect = {
            "effect": "invitation",
            "slot_id": observation["slot_id"],
            "meeting_id": f"meeting-{agent}",
            "idempotency_key": idempotency_key,
        }
        if not guarded or not any(x["idempotency_key"] == idempotency_key for x in self.invitations):
            self.invitations.append(effect)
        return {"booking_ids": list(state["booking_ids"]), "version": state["version"]}

    def score(self) -> OracleResult:
        invariant = [key for key, state in self.slots.items() if len(state["booking_ids"]) > 1]
        invite_counts = {key: 0 for key in self.slots}
        for effect in self.invitations:
            invite_counts[effect["slot_id"]] += 1
        side = [key for key, count in invite_counts.items() if count > 1]
        lost = [
            key
            for key, count in invite_counts.items()
            if count != len(self.slots[key]["booking_ids"])
        ]
        return OracleResult(invariant, side, lost, [])

    def final_state(self) -> dict[str, Any]:
        return {"slots": self.slots, "invitations": self.invitations}

    def successful_action_capacity(self) -> int:
        return len(self.slots)

    def side_effect_count(self) -> int:
        return len(self.invitations)


class SupportEnvironment(DomainEnvironment):
    name = "support"

    def __init__(self, agent_count: int, contention: int, scenario_seed: int):
        super().__init__(agent_count, contention, scenario_seed)
        self.refund_cents = [2500, 5000, 7500][scenario_seed % 3]
        self.cases: dict[str, dict[str, int]] = {}
        self.refunds: list[dict[str, Any]] = []
        self.notifications: list[dict[str, Any]] = []
        for agent in range(agent_count):
            case_id = assigned_key("case", agent, agent_count, contention)
            self.cases.setdefault(
                case_id,
                {"eligible_cents": self.refund_cents, "refunded_cents": 0, "version": 0},
            )

    def _key(self, agent: int) -> str:
        return assigned_key("case", agent, self.agent_count, self.contention)

    def observe(self, agent: int) -> dict[str, Any]:
        case_id = self._key(agent)
        state = self.cases[case_id]
        return {
            "resource_id": case_id,
            "case_id": case_id,
            "eligible_cents": state["eligible_cents"],
            "refunded_cents": state["refunded_cents"],
            "capacity": state["eligible_cents"],
            "allocated": state["refunded_cents"],
            "version": state["version"],
        }

    def task(self, agent: int) -> dict[str, Any]:
        return {
            "action": "issue_refund",
            "amount": self.refund_cents,
            "domain": self.name,
            "case_id": self._key(agent),
        }

    def is_stale(self, observation: dict[str, Any]) -> bool:
        return self.cases[observation["case_id"]]["version"] != observation["version"]

    def can_apply(self, observation: dict[str, Any], amount: int) -> bool:
        return observation["refunded_cents"] + amount <= observation["eligible_cents"]

    def commit(self, agent, observation, amount, stale_write, idempotency_key, guarded):
        state = self.cases[observation["case_id"]]
        candidate = (
            observation["refunded_cents"] + amount
            if stale_write
            else state["refunded_cents"] + amount
        )
        if guarded and candidate > state["eligible_cents"]:
            raise ValueError("refund_eligibility_invariant")
        state["refunded_cents"] = candidate
        state["version"] += 1
        refund = {
            "effect": "refund",
            "case_id": observation["case_id"],
            "amount_cents": amount,
            "idempotency_key": idempotency_key,
        }
        notice = {
            "effect": "refund_notification",
            "case_id": observation["case_id"],
            "idempotency_key": idempotency_key,
        }
        if not guarded or not any(x["idempotency_key"] == idempotency_key for x in self.refunds):
            self.refunds.append(refund)
            self.notifications.append(notice)
        return dict(state)

    def score(self) -> OracleResult:
        invariant = [
            key
            for key, state in self.cases.items()
            if state["refunded_cents"] > state["eligible_cents"]
        ]
        effect_totals = {key: 0 for key in self.cases}
        effect_counts = {key: 0 for key in self.cases}
        for effect in self.refunds:
            effect_totals[effect["case_id"]] += effect["amount_cents"]
            effect_counts[effect["case_id"]] += 1
        side = [
            key
            for key, total in effect_totals.items()
            if total > self.cases[key]["eligible_cents"]
            or effect_counts[key] != sum(n["case_id"] == key for n in self.notifications)
        ]
        lost = [
            key
            for key, total in effect_totals.items()
            if total != self.cases[key]["refunded_cents"]
        ]
        return OracleResult(invariant, side, lost, [])

    def final_state(self) -> dict[str, Any]:
        return {"cases": self.cases, "refunds": self.refunds, "notifications": self.notifications}

    def successful_action_capacity(self) -> int:
        return len(self.cases)

    def side_effect_count(self) -> int:
        return len(self.refunds)


class CloudEnvironment(DomainEnvironment):
    name = "cloud"

    def __init__(self, agent_count: int, contention: int, scenario_seed: int):
        super().__init__(agent_count, contention, scenario_seed)
        self.pools: dict[str, dict[str, Any]] = {}
        self.deployments: list[dict[str, Any]] = []
        self.dependency_generation = 10 + scenario_seed
        for agent in range(agent_count):
            pool = assigned_key("quota", agent, agent_count, contention)
            self.pools.setdefault(
                pool,
                {
                    "quota_units": 1,
                    "used_units": 0,
                    "dependency_ready": True,
                    "dependency_generation": self.dependency_generation,
                    "version": 0,
                },
            )

    def _key(self, agent: int) -> str:
        return assigned_key("quota", agent, self.agent_count, self.contention)

    def observe(self, agent: int) -> dict[str, Any]:
        pool = self._key(agent)
        state = self.pools[pool]
        return {
            "resource_id": pool,
            "pool_id": pool,
            "quota_units": state["quota_units"],
            "used_units": state["used_units"],
            "dependency_ready": state["dependency_ready"],
            "dependency_generation": state["dependency_generation"],
            "capacity": state["quota_units"],
            "allocated": state["used_units"],
            "version": state["version"],
        }

    def task(self, agent: int) -> dict[str, Any]:
        return {
            "action": "deploy_service",
            "amount": 1,
            "domain": self.name,
            "deployment_id": f"deployment-{agent}",
        }

    def is_stale(self, observation: dict[str, Any]) -> bool:
        return self.pools[observation["pool_id"]]["version"] != observation["version"]

    def can_apply(self, observation: dict[str, Any], amount: int) -> bool:
        return bool(observation["dependency_ready"]) and (
            observation["used_units"] + amount <= observation["quota_units"]
        )

    def commit(self, agent, observation, amount, stale_write, idempotency_key, guarded):
        state = self.pools[observation["pool_id"]]
        candidate = observation["used_units"] + amount if stale_write else state["used_units"] + amount
        if guarded and (
            not state["dependency_ready"] or candidate > state["quota_units"]
        ):
            raise ValueError("cloud_deployment_invariant")
        state["used_units"] = candidate
        state["version"] += 1
        effect = {
            "effect": "deployment",
            "pool_id": observation["pool_id"],
            "deployment_id": f"deployment-{agent}",
            "dependency_generation": observation["dependency_generation"],
            "idempotency_key": idempotency_key,
        }
        if not guarded or not any(x["idempotency_key"] == idempotency_key for x in self.deployments):
            self.deployments.append(effect)
        return dict(state)

    def score(self) -> OracleResult:
        invariant = [
            key for key, state in self.pools.items() if state["used_units"] > state["quota_units"]
        ]
        deployment_counts = {key: 0 for key in self.pools}
        for effect in self.deployments:
            deployment_counts[effect["pool_id"]] += 1
        side = [
            key
            for key, count in deployment_counts.items()
            if count > self.pools[key]["quota_units"]
        ]
        lost = [
            key
            for key, count in deployment_counts.items()
            if count != self.pools[key]["used_units"]
        ]
        dependency = [
            effect["deployment_id"]
            for effect in self.deployments
            if effect["dependency_generation"]
            != self.pools[effect["pool_id"]]["dependency_generation"]
        ]
        return OracleResult(invariant, side, lost, dependency)

    def final_state(self) -> dict[str, Any]:
        return {"quota_pools": self.pools, "deployments": self.deployments}

    def successful_action_capacity(self) -> int:
        return sum(state["quota_units"] for state in self.pools.values())

    def side_effect_count(self) -> int:
        return len(self.deployments)


DOMAIN_TYPES: dict[str, type[DomainEnvironment]] = {
    "inventory": InventoryEnvironment,
    "calendar": CalendarEnvironment,
    "support": SupportEnvironment,
    "cloud": CloudEnvironment,
}


def build_environment(
    domain: str, agent_count: int, contention: int, scenario_seed: int = 0
) -> DomainEnvironment:
    try:
        domain_type = DOMAIN_TYPES[domain]
    except KeyError as exc:
        raise ValueError(f"Unknown domain: {domain}") from exc
    return domain_type(agent_count, contention, scenario_seed)
