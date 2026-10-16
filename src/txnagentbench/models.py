from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class Decision:
    action: str
    amount: int
    raw_output: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0


class DecisionModel(Protocol):
    def decide(self, observation: dict[str, Any], task: dict[str, Any]) -> Decision: ...
    def close(self) -> None: ...


class ScriptedDecisionModel:
    """Deterministic agent used only to validate schedules, policies, and oracles."""

    def decide(self, observation: dict[str, Any], task: dict[str, Any]) -> Decision:
        available = int(observation["capacity"]) - int(observation["allocated"])
        action = "apply" if available >= int(task["amount"]) else "reject"
        return Decision(action=action, amount=int(task["amount"]), raw_output=action)

    def close(self) -> None:
        return None


class TransformersDecisionModel:
    """Small JSON-only inference adapter suitable for one A100 Colab runtime."""

    def __init__(self, model_id: str, revision: str | None = None, quantization: str = "bf16"):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("Install the GPU extras with: pip install -e '.[gpu]'") from exc

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, trust_remote_code=False
        )
        kwargs: dict[str, Any] = {
            "revision": revision,
            "trust_remote_code": False,
            "device_map": "auto",
        }
        if quantization == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            kwargs["dtype"] = torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    def decide(self, observation: dict[str, Any], task: dict[str, Any]) -> Decision:
        system = (
            "You are an agent completing a resource-allocation task from one observed snapshot. "
            "If allocated + requested amount is less than or equal to capacity, choose apply. "
            "Otherwise choose reject. Use only the supplied observation; do not speculate about "
            "other agents or future changes. Return exactly one JSON object with action and amount, "
            "for example {\"action\":\"apply\",\"amount\":1}. Valid actions are apply and reject. "
            "Do not include reasoning, Markdown, or additional text."
        )
        user = json.dumps(
            {
                "observation": observation,
                "requested_amount": int(task["amount"]),
                "domain": task["domain"],
                "instruction": "Fulfil the request when the observed capacity is sufficient.",
            },
            sort_keys=True,
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = system + "\n" + user + "\nJSON:"
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        start = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(
                **encoded,
                max_new_tokens=48,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        latency = time.perf_counter() - start
        new_tokens = output[0, encoded["input_ids"].shape[1] :]
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        match = re.search(r"\{.*?\}", raw, flags=re.DOTALL)
        try:
            payload = json.loads(match.group(0) if match else raw)
            action = str(payload.get("action", "invalid")).lower()
            amount = int(payload.get("amount", task["amount"]))
        except (ValueError, TypeError, json.JSONDecodeError):
            action, amount = "invalid", 0
        if action not in {"apply", "reject"}:
            action = "invalid"
        return Decision(
            action=action,
            amount=amount,
            raw_output=raw,
            input_tokens=int(encoded["input_ids"].numel()),
            output_tokens=int(new_tokens.numel()),
            latency_seconds=latency,
        )

    def close(self) -> None:
        del self.model
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def build_model(config: dict[str, Any]) -> DecisionModel:
    provider = config.get("provider", "scripted")
    if provider == "scripted":
        return ScriptedDecisionModel()
    if provider == "transformers":
        return TransformersDecisionModel(
            model_id=config["model_id"],
            revision=config.get("revision"),
            quantization=config.get("quantization", "bf16"),
        )
    raise ValueError(f"Unknown model provider: {provider}")

