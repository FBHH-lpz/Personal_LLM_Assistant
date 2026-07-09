"""Abstract base classes for LLM providers and embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional, Union


@dataclass
class ChatResponse:
    """A completed chat response."""
    content: str
    model: str = ""
    usage: dict = field(default_factory=dict)


@dataclass
class ChatChunk:
    """A streaming delta chunk."""
    delta_content: str
    finish_reason: Optional[str] = None


@dataclass
class EmbeddingResponse:
    """Batch embedding result."""
    embeddings: list[list[float]]
    model: str = ""
    usage: dict = field(default_factory=dict)


class ChatModel(ABC):
    """Unified chat completion interface for all LLM providers."""

    model_id: str
    base_url: str
    api_key: str

    @abstractmethod
    async def achat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False,
    ) -> Union[ChatResponse, AsyncIterator[ChatChunk]]:
        """Send messages and get a response (or stream of chunks)."""
        ...


class EmbeddingProvider(ABC):
    """Unified embedding interface."""

    model_id: str
    base_url: str
    api_key: str

    @abstractmethod
    async def aembed(
        self,
        texts: list[str],
    ) -> EmbeddingResponse:
        """Generate embeddings for a list of texts."""
        ...

    async def aembed_single(self, text: str) -> list[float]:
        """Convenience: embed a single string."""
        resp = await self.aembed([text])
        return resp.embeddings[0]
