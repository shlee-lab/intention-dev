"""Abstract base class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from imaro.models.schemas import LLMResponse


class LLMProvider(ABC):
    """Base class all LLM providers must implement."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send a prompt and return the model's response."""
        ...

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model identifier string."""
        ...
