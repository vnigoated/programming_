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
class CapacityProfile:
    family: str
    capacity: int
    initial: int
    request: int


def capacity_profile(scenario_seed: int) -> CapacityProfile:
    profiles = (
        CapacityProfile("scarce", capacity=1, initial=0, request=1),
        CapacityProfile("two-fit", capacity=2, initial=0, request=1),
        CapacityProfile("partially-allocated", capacity=3, initial=1, request=2),
        CapacityProfile("exhausted", capacity=1, initial=1, request=1),
    )
    return profiles[scenario_seed % len(profiles)]


def assigned_agent_counts(prefix: str, agent_count: int, contention: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for agent in range(agent_count):
        key = assigned_key(prefix, agent, agent_count, contention)
        counts[key] = counts.get(key, 0) + 1
    return counts


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

    @abstractmethod
    def replay_effect(self, idempotency_key: str, guarded: bool) -> bool:
        """Replay an at-least-once effect. Return true when another effect is delivered."""
        ...

    def inject_fault(self, fault_mode: str) -> list[dict[str, Any]]:
        return []


class InventoryEnvironment(DomainEnvironment):
    name = "inventory"

    def __init__(self, agent_count: int, contention: int, scenario_seed: int):
        super().__init__(agent_count, contention, scenario_seed)
        self.profile = capacity_profile(scenario_seed)
        self.agent_counts = assigned_agent_counts("sku", agent_count, contention)
        self.items: dict[str, dict[str, int]] = {}
        self.shipments: list[dict[str, Any]] = []
        for agent in range(agent_count):
            sku = assigned_key("sku", agent, agent_count, contention)
            self.items.setdefault(
                sku,
                {
                    "stock": self.profile.capacity,
                    "reserved": self.profile.initial,
                    "version": 0,
                },
            )
        for sku in self.items:
            if self.profile.initial:
                self.shipments.append(
                    {
                        "effect": "shipment",
                        "sku": sku,
                        "order_id": f"baseline-{sku}",
                        "amount": self.profile.initial,
                        "idempotency_key": f"baseline:{sku}",
                        "baseline": True,
                    }
                )

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
            "available": state["stock"] - state["reserved"],
            "scenario_family": self.profile.family,
            "version": state["version"],
        }

    def task(self, agent: int) -> dict[str, Any]:
        return {
            "action": "reserve_inventory",
            "amount": self.profile.request,
            "domain": self.name,
            "order_id": f"order-{agent}",
            "scenario_family": self.profile.family,
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
            "amount": amount,
            "idempotency_key": idempotency_key,
            "baseline": False,
        }
        if not guarded or not any(x["idempotency_key"] == idempotency_key for x in self.shipments):
            self.shipments.append(effect)
        return dict(state)

    def score(self) -> OracleResult:
        invariant = [key for key, state in self.items.items() if state["reserved"] > state["stock"]]
        effect_totals = {key: 0 for key in self.items}
        for effect in self.shipments:
            effect_totals[effect["sku"]] += effect["amount"]
        side = [key for key, total in effect_totals.items() if total > self.items[key]["stock"]]
        lost = [key for key, total in effect_totals.items() if total != self.items[key]["reserved"]]
        return OracleResult(invariant, side, lost, [])

    def final_state(self) -> dict[str, Any]:
        return {"items": self.items, "shipments": self.shipments}

    def successful_action_capacity(self) -> int:
        return sum(
            min(
                self.agent_counts[key],
                (state["stock"] - self.profile.initial) // self.profile.request,
            )
            for key, state in self.items.items()
        )

    def side_effect_count(self) -> int:
        return sum(not effect.get("baseline", False) for effect in self.shipments)

    def replay_effect(self, idempotency_key: str, guarded: bool) -> bool:
        effect = next(x for x in self.shipments if x["idempotency_key"] == idempotency_key)
        if guarded and any(x["idempotency_key"] == idempotency_key for x in self.shipments):
            return False
        self.shipments.append(dict(effect))
        return True


class CalendarEnvironment(DomainEnvironment):
    name = "calendar"

    def __init__(self, agent_count: int, contention: int, scenario_seed: int):
        super().__init__(agent_count, contention, scenario_seed)
        self.scenario_family = (
            "prebooked" if scenario_seed % 4 == 3 else ("long-meeting" if scenario_seed % 4 == 2 else "open-slot")
        )
        self.duration_minutes = (30, 45, 60, 30)[scenario_seed % 4]
        self.agent_counts = assigned_agent_counts(
            f"room-{9 + scenario_seed % 8:02d}00", agent_count, contention
        )
        self.slots: dict[str, dict[str, Any]] = {}
        self.initial_bookings: dict[str, int] = {}
        self.invitations: list[dict[str, Any]] = []
        hour = 9 + scenario_seed % 8
        for agent in range(agent_count):
            slot = assigned_key(f"room-{hour:02d}00", agent, agent_count, contention)
            initial = [f"baseline-{slot}"] if self.scenario_family == "prebooked" else []
            self.slots.setdefault(slot, {"booking_ids": initial, "version": 0})
            self.initial_bookings[slot] = len(initial)
        for slot, state in self.slots.items():
            if state["booking_ids"]:
                self.invitations.append(
                    {
                        "effect": "invitation",
                        "slot_id": slot,
                        "meeting_id": state["booking_ids"][0],
                        "idempotency_key": f"baseline:{slot}",
                        "baseline": True,
                    }
                )

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
            "available": 1 - len(state["booking_ids"]),
            "duration_minutes": self.duration_minutes,
            "scenario_family": self.scenario_family,
            "version": state["version"],
        }

    def task(self, agent: int) -> dict[str, Any]:
        return {
            "action": "book_meeting",
            "amount": 1,
            "domain": self.name,
            "meeting_id": f"meeting-{agent}",
            "duration_minutes": self.duration_minutes,
            "scenario_family": self.scenario_family,
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
            "baseline": False,
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
        return sum(
            min(self.agent_counts[key], 1 - self.initial_bookings[key])
            for key, state in self.slots.items()
        )

    def side_effect_count(self) -> int:
        return sum(not effect.get("baseline", False) for effect in self.invitations)

    def replay_effect(self, idempotency_key: str, guarded: bool) -> bool:
        effect = next(x for x in self.invitations if x["idempotency_key"] == idempotency_key)
        if guarded and any(x["idempotency_key"] == idempotency_key for x in self.invitations):
            return False
        self.invitations.append(dict(effect))
        return True


class SupportEnvironment(DomainEnvironment):
    name = "support"

    def __init__(self, agent_count: int, contention: int, scenario_seed: int):
        super().__init__(agent_count, contention, scenario_seed)
        self.profile = capacity_profile(scenario_seed)
        self.unit_cents = 2500
        self.refund_cents = self.profile.request * self.unit_cents
        self.eligible_cents = self.profile.capacity * self.unit_cents
        self.initial_refunded_cents = self.profile.initial * self.unit_cents
        self.agent_counts = assigned_agent_counts("case", agent_count, contention)
        self.cases: dict[str, dict[str, int]] = {}
        self.refunds: list[dict[str, Any]] = []
        self.notifications: list[dict[str, Any]] = []
        for agent in range(agent_count):
            case_id = assigned_key("case", agent, agent_count, contention)
            self.cases.setdefault(
                case_id,
                {
                    "eligible_cents": self.eligible_cents,
                    "refunded_cents": self.initial_refunded_cents,
                    "version": 0,
                },
            )
        for case_id in self.cases:
            if self.initial_refunded_cents:
                key = f"baseline:{case_id}"
                self.refunds.append(
                    {
                        "effect": "refund",
                        "case_id": case_id,
                        "amount_cents": self.initial_refunded_cents,
                        "idempotency_key": key,
                        "baseline": True,
                    }
                )
                self.notifications.append(
                    {
                        "effect": "refund_notification",
                        "case_id": case_id,
                        "idempotency_key": key,
                        "baseline": True,
                    }
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
            "available": state["eligible_cents"] - state["refunded_cents"],
            "scenario_family": self.profile.family,
            "version": state["version"],
        }

    def task(self, agent: int) -> dict[str, Any]:
        return {
            "action": "issue_refund",
            "amount": self.refund_cents,
            "domain": self.name,
            "case_id": self._key(agent),
            "scenario_family": self.profile.family,
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
            "baseline": False,
        }
        notice = {
            "effect": "refund_notification",
            "case_id": observation["case_id"],
            "idempotency_key": idempotency_key,
            "baseline": False,
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
        return sum(
            min(
                self.agent_counts[key],
                (state["eligible_cents"] - self.initial_refunded_cents) // self.refund_cents,
            )
            for key, state in self.cases.items()
        )

    def side_effect_count(self) -> int:
        return sum(not effect.get("baseline", False) for effect in self.refunds)

    def replay_effect(self, idempotency_key: str, guarded: bool) -> bool:
        refund = next(x for x in self.refunds if x["idempotency_key"] == idempotency_key)
        notice = next(x for x in self.notifications if x["idempotency_key"] == idempotency_key)
        if guarded and any(x["idempotency_key"] == idempotency_key for x in self.refunds):
            return False
        self.refunds.append(dict(refund))
        self.notifications.append(dict(notice))
        return True


class CloudEnvironment(DomainEnvironment):
    name = "cloud"

    def __init__(self, agent_count: int, contention: int, scenario_seed: int):
        super().__init__(agent_count, contention, scenario_seed)
        self.profile = capacity_profile(scenario_seed)
        self.agent_counts = assigned_agent_counts("quota", agent_count, contention)
        self.pools: dict[str, dict[str, Any]] = {}
        self.deployments: list[dict[str, Any]] = []
        self.dependency_generation = 10 + scenario_seed
        for agent in range(agent_count):
            pool = assigned_key("quota", agent, agent_count, contention)
            self.pools.setdefault(
                pool,
                {
                    "quota_units": self.profile.capacity,
                    "used_units": self.profile.initial,
                    "dependency_ready": self.profile.family != "exhausted",
                    "dependency_generation": self.dependency_generation,
                    "version": 0,
                },
            )
        for pool in self.pools:
            if self.profile.initial:
                self.deployments.append(
                    {
                        "effect": "deployment",
                        "pool_id": pool,
                        "deployment_id": f"baseline-{pool}",
                        "amount_units": self.profile.initial,
                        "dependency_generation": self.dependency_generation,
                        "idempotency_key": f"baseline:{pool}",
                        "baseline": True,
                    }
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
            "available": state["quota_units"] - state["used_units"],
            "scenario_family": self.profile.family,
            "version": state["version"],
        }

    def task(self, agent: int) -> dict[str, Any]:
        return {
            "action": "deploy_service",
            "amount": self.profile.request,
            "domain": self.name,
            "deployment_id": f"deployment-{agent}",
            "scenario_family": self.profile.family,
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
            "amount_units": amount,
            "idempotency_key": idempotency_key,
            "baseline": False,
        }
        if not guarded or not any(x["idempotency_key"] == idempotency_key for x in self.deployments):
            self.deployments.append(effect)
        return dict(state)

    def score(self) -> OracleResult:
        invariant = [
            key for key, state in self.pools.items() if state["used_units"] > state["quota_units"]
        ]
        deployment_totals = {key: 0 for key in self.pools}
        for effect in self.deployments:
            deployment_totals[effect["pool_id"]] += effect["amount_units"]
        side = [
            key
            for key, total in deployment_totals.items()
            if total > self.pools[key]["quota_units"]
        ]
        lost = [
            key
            for key, total in deployment_totals.items()
            if total != self.pools[key]["used_units"]
        ]
        dependency = [
            effect["deployment_id"]
            for effect in self.deployments
            if not effect.get("baseline", False)
            and effect["dependency_generation"]
            != self.pools[effect["pool_id"]]["dependency_generation"]
        ]
        return OracleResult(invariant, side, lost, dependency)

    def final_state(self) -> dict[str, Any]:
        return {"quota_pools": self.pools, "deployments": self.deployments}

    def successful_action_capacity(self) -> int:
        if self.profile.family == "exhausted":
            return 0
        return sum(
            min(
                self.agent_counts[key],
                (state["quota_units"] - self.profile.initial) // self.profile.request,
            )
            for key, state in self.pools.items()
        )

    def side_effect_count(self) -> int:
        return sum(not effect.get("baseline", False) for effect in self.deployments)

    def replay_effect(self, idempotency_key: str, guarded: bool) -> bool:
        effect = next(x for x in self.deployments if x["idempotency_key"] == idempotency_key)
        if guarded and any(x["idempotency_key"] == idempotency_key for x in self.deployments):
            return False
        self.deployments.append(dict(effect))
        return True

    def inject_fault(self, fault_mode: str) -> list[dict[str, Any]]:
        if fault_mode != "dependency_change":
            return []
        events = []
        for pool_id, state in self.pools.items():
            state["dependency_generation"] += 1
            state["version"] += 1
            events.append(
                {
                    "event": "dependency_change",
                    "resource_id": pool_id,
                    "dependency_generation": state["dependency_generation"],
                    "version": state["version"],
                }
            )
        return events


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
