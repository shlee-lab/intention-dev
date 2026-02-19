"""Usage tracking for LLM API calls."""

from __future__ import annotations

from dataclasses import dataclass, field

from imaro.models.pricing import calculate_cost
from imaro.models.schemas import LLMResponse
from imaro.providers.base import LLMProvider


@dataclass
class CallRecord:
    """A single LLM API call record."""

    role: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float


@dataclass
class RoleBreakdown:
    """Aggregated usage for a single role."""

    role: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0


@dataclass
class UsageSummary:
    """Aggregated usage across all roles."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    total_calls: int = 0
    by_role: list[RoleBreakdown] = field(default_factory=list)


class UsageTracker:
    """Accumulates LLM usage across all API calls."""

    def __init__(self) -> None:
        self._records: list[CallRecord] = []

    def record(self, role: str, response: LLMResponse) -> None:
        """Record usage from an LLM response."""
        input_tokens = response.usage.get("input_tokens", 0)
        output_tokens = response.usage.get("output_tokens", 0)
        cost = calculate_cost(response.model, input_tokens, output_tokens)
        self._records.append(
            CallRecord(
                role=role,
                model=response.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
            )
        )

    def summary(self) -> UsageSummary:
        """Compute aggregated usage summary."""
        role_map: dict[str, RoleBreakdown] = {}
        total_in = 0
        total_out = 0
        total_cost = 0.0

        for rec in self._records:
            total_in += rec.input_tokens
            total_out += rec.output_tokens
            total_cost += rec.cost

            if rec.role not in role_map:
                role_map[rec.role] = RoleBreakdown(role=rec.role)
            rb = role_map[rec.role]
            rb.calls += 1
            rb.input_tokens += rec.input_tokens
            rb.output_tokens += rec.output_tokens
            rb.cost += rec.cost

        return UsageSummary(
            total_input_tokens=total_in,
            total_output_tokens=total_out,
            total_cost=total_cost,
            total_calls=len(self._records),
            by_role=sorted(role_map.values(), key=lambda r: r.role),
        )


class TrackedProvider(LLMProvider):
    """Wraps any LLMProvider to automatically track usage."""

    def __init__(self, inner: LLMProvider, tracker: UsageTracker, role: str) -> None:
        self._inner = inner
        self._tracker = tracker
        self._role = role

    async def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        resp = await self._inner.generate(
            prompt, system=system, temperature=temperature, max_tokens=max_tokens
        )
        self._tracker.record(self._role, resp)
        return resp

    def get_model_name(self) -> str:
        return self._inner.get_model_name()
