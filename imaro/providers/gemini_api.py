"""Gemini API provider using the google-genai SDK."""

from __future__ import annotations

import asyncio
import logging
import os

from google import genai
from google.genai import types

from imaro.models.schemas import LLMResponse
from imaro.providers.base import LLMProvider

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.5-flash"
MAX_RETRIES = 3
BASE_DELAY = 1.0  # seconds


class GeminiAPIProvider(LLMProvider):
    """LLM provider backed by the Google Gemini API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        self._model = model
        self._client = genai.Client(api_key=self._api_key)

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
                config = types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    system_instruction=system or None,
                )

                response = await self._client.aio.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=config,
                )

                content = response.text or ""

                usage: dict = {}
                if response.usage_metadata:
                    usage = {
                        "input_tokens": response.usage_metadata.prompt_token_count,
                        "output_tokens": response.usage_metadata.candidates_token_count,
                    }

                return LLMResponse(
                    content=content,
                    model=self._model,
                    usage=usage,
                )

            except Exception as exc:
                last_error = exc
                err_str = str(exc).lower()
                is_retryable = (
                    "429" in err_str
                    or "rate" in err_str
                    or "500" in err_str
                    or "503" in err_str
                    or "overloaded" in err_str
                    or "unavailable" in err_str
                )

                if is_retryable and attempt < MAX_RETRIES:
                    delay = BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning(
                        "Gemini API error (attempt %d/%d), retrying in %.1fs: %s",
                        attempt,
                        MAX_RETRIES,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)
                elif not is_retryable:
                    raise

        raise RuntimeError(
            f"Gemini API call failed after {MAX_RETRIES} retries: {last_error}"
        ) from last_error
