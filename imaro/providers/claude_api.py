"""Claude API provider using the Anthropic SDK."""

from __future__ import annotations

import asyncio
import logging
import os

import anthropic

from imaro.models.schemas import LLMResponse
from imaro.providers.base import LLMProvider

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-20250514"
MAX_RETRIES = 3
BASE_DELAY = 1.0  # seconds


class ClaudeAPIProvider(LLMProvider):
    """LLM provider backed by the Anthropic Messages API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._model = model
        self._client = anthropic.AsyncAnthropic(api_key=self._api_key)

    def get_model_name(self) -> str:
        return self._model

    async def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        last_error: Exception | None = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                kwargs: dict = {
                    "model": self._model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": prompt}],
                }
                if system:
                    kwargs["system"] = system

                response = await self._client.messages.create(**kwargs)

                content = ""
                for block in response.content:
                    if block.type == "text":
                        content += block.text

                usage = {}
                if response.usage:
                    usage = {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                    }

                return LLMResponse(
                    content=content,
                    model=response.model,
                    usage=usage,
                )

            except anthropic.RateLimitError as exc:
                last_error = exc
                delay = BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "Rate limited (attempt %d/%d), retrying in %.1fs",
                    attempt,
                    MAX_RETRIES,
                    delay,
                )
                await asyncio.sleep(delay)

            except anthropic.APIStatusError as exc:
                last_error = exc
                if exc.status_code >= 500:
                    delay = BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning(
                        "Server error %d (attempt %d/%d), retrying in %.1fs",
                        exc.status_code,
                        attempt,
                        MAX_RETRIES,
                        delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    raise

        raise RuntimeError(
            f"LLM call failed after {MAX_RETRIES} retries: {last_error}"
        ) from last_error
