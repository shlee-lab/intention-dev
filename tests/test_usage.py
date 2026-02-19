"""Tests for pricing, UsageTracker, and TrackedProvider."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from imaro.models.pricing import MODEL_PRICING, calculate_cost
from imaro.models.schemas import LLMResponse
from imaro.models.usage import (
    CallRecord,
    TrackedProvider,
    UsageSummary,
    UsageTracker,
)
from imaro.providers.base import LLMProvider


# ── Pricing ──────────────────────────────────────────────────────────────────


class TestCalculateCost:
    def test_known_model(self):
        # claude-sonnet-4: $3/1M in, $15/1M out
        cost = calculate_cost("claude-sonnet-4-20250514", 1_000_000, 1_000_000)
        assert cost == pytest.approx(18.00)

    def test_known_model_partial_tokens(self):
        cost = calculate_cost("claude-sonnet-4-20250514", 500, 100)
        expected = (500 / 1_000_000) * 3.00 + (100 / 1_000_000) * 15.00
        assert cost == pytest.approx(expected)

    def test_unknown_model_returns_zero(self):
        assert calculate_cost("nonexistent-model", 1000, 1000) == 0.0

    def test_zero_tokens(self):
        assert calculate_cost("claude-sonnet-4-20250514", 0, 0) == 0.0

    def test_gemini_flash(self):
        # gemini-2.5-flash: $0.15/1M in, $0.60/1M out
        cost = calculate_cost("gemini-2.5-flash", 2_000_000, 500_000)
        expected = (2_000_000 / 1_000_000) * 0.15 + (500_000 / 1_000_000) * 0.60
        assert cost == pytest.approx(expected)

    def test_all_models_in_pricing_table(self):
        """Smoke test: every model in the table produces a non-negative cost."""
        for model in MODEL_PRICING:
            cost = calculate_cost(model, 1000, 1000)
            assert cost >= 0.0


# ── UsageTracker ─────────────────────────────────────────────────────────────


class TestUsageTracker:
    def test_empty_tracker(self):
        tracker = UsageTracker()
        s = tracker.summary()
        assert s.total_calls == 0
        assert s.total_input_tokens == 0
        assert s.total_output_tokens == 0
        assert s.total_cost == 0.0
        assert s.by_role == []

    def test_single_record(self):
        tracker = UsageTracker()
        resp = LLMResponse(
            content="hello",
            model="claude-sonnet-4-20250514",
            usage={"input_tokens": 100, "output_tokens": 50},
        )
        tracker.record("refiner", resp)
        s = tracker.summary()
        assert s.total_calls == 1
        assert s.total_input_tokens == 100
        assert s.total_output_tokens == 50
        expected_cost = calculate_cost("claude-sonnet-4-20250514", 100, 50)
        assert s.total_cost == pytest.approx(expected_cost)
        assert len(s.by_role) == 1
        assert s.by_role[0].role == "refiner"
        assert s.by_role[0].calls == 1

    def test_multiple_roles(self):
        tracker = UsageTracker()
        resp1 = LLMResponse(
            content="a", model="gemini-2.5-flash",
            usage={"input_tokens": 200, "output_tokens": 100},
        )
        resp2 = LLMResponse(
            content="b", model="gemini-2.5-flash",
            usage={"input_tokens": 300, "output_tokens": 150},
        )
        resp3 = LLMResponse(
            content="c", model="claude-sonnet-4-20250514",
            usage={"input_tokens": 400, "output_tokens": 200},
        )
        tracker.record("planner", resp1)
        tracker.record("planner", resp2)
        tracker.record("reviewer", resp3)

        s = tracker.summary()
        assert s.total_calls == 3
        assert s.total_input_tokens == 900
        assert s.total_output_tokens == 450
        assert len(s.by_role) == 2  # planner and reviewer
        # by_role is sorted by role name
        planner = next(r for r in s.by_role if r.role == "planner")
        assert planner.calls == 2
        assert planner.input_tokens == 500
        assert planner.output_tokens == 250

    def test_unknown_model_zero_cost(self):
        tracker = UsageTracker()
        resp = LLMResponse(
            content="x", model="unknown-model-v9",
            usage={"input_tokens": 1000, "output_tokens": 500},
        )
        tracker.record("test", resp)
        s = tracker.summary()
        assert s.total_cost == 0.0
        assert s.total_input_tokens == 1000

    def test_missing_usage_keys(self):
        tracker = UsageTracker()
        resp = LLMResponse(content="x", model="gemini-2.5-flash", usage={})
        tracker.record("test", resp)
        s = tracker.summary()
        assert s.total_input_tokens == 0
        assert s.total_output_tokens == 0


# ── TrackedProvider ──────────────────────────────────────────────────────────


class TestTrackedProvider:
    @pytest.mark.asyncio
    async def test_generate_delegates_and_tracks(self):
        inner = AsyncMock(spec=LLMProvider)
        inner.generate.return_value = LLMResponse(
            content="result",
            model="gemini-2.5-flash",
            usage={"input_tokens": 50, "output_tokens": 25},
        )
        inner.get_model_name.return_value = "gemini-2.5-flash"

        tracker = UsageTracker()
        wrapped = TrackedProvider(inner, tracker, "reviewer")

        resp = await wrapped.generate("prompt", system="sys", temperature=0.5, max_tokens=2048)

        assert resp.content == "result"
        inner.generate.assert_awaited_once_with(
            "prompt", system="sys", temperature=0.5, max_tokens=2048
        )
        s = tracker.summary()
        assert s.total_calls == 1
        assert s.by_role[0].role == "reviewer"

    def test_get_model_name_delegates(self):
        inner = AsyncMock(spec=LLMProvider)
        inner.get_model_name.return_value = "test-model"
        tracker = UsageTracker()
        wrapped = TrackedProvider(inner, tracker, "test")
        assert wrapped.get_model_name() == "test-model"

    @pytest.mark.asyncio
    async def test_multiple_calls_accumulate(self):
        inner = AsyncMock(spec=LLMProvider)
        inner.generate.return_value = LLMResponse(
            content="ok",
            model="gemini-2.0-flash",
            usage={"input_tokens": 100, "output_tokens": 50},
        )

        tracker = UsageTracker()
        wrapped = TrackedProvider(inner, tracker, "planner")

        await wrapped.generate("p1")
        await wrapped.generate("p2")
        await wrapped.generate("p3")

        s = tracker.summary()
        assert s.total_calls == 3
        assert s.total_input_tokens == 300
        assert s.total_output_tokens == 150
