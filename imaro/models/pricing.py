"""Model pricing table and cost calculation."""

from __future__ import annotations

# Prices per 1 million tokens (USD)
MODEL_PRICING: dict[str, dict[str, float]] = {
    # Anthropic
    "claude-sonnet-4-20250514": {"input_per_1m": 3.00, "output_per_1m": 15.00},
    "claude-opus-4-20250514": {"input_per_1m": 15.00, "output_per_1m": 75.00},
    "claude-haiku-4-20250514": {"input_per_1m": 0.80, "output_per_1m": 4.00},
    # Google Gemini
    "gemini-2.5-flash": {"input_per_1m": 0.15, "output_per_1m": 0.60},
    "gemini-2.5-pro": {"input_per_1m": 1.25, "output_per_1m": 10.00},
    "gemini-2.0-flash": {"input_per_1m": 0.10, "output_per_1m": 0.40},
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate the USD cost for a given model and token counts.

    Returns 0.0 if the model is not in the pricing table.
    """
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        return 0.0
    input_cost = (input_tokens / 1_000_000) * pricing["input_per_1m"]
    output_cost = (output_tokens / 1_000_000) * pricing["output_per_1m"]
    return input_cost + output_cost
