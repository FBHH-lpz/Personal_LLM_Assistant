"""OpenAI-compatible HTTP client shared by Tongyi, DeepSeek, Zhipu.

All three providers expose chat/completions with slight differences in
field naming (e.g. Tongyi uses ``output``, DeepSeek uses ``choices``).
This module handles the normalization.
"""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator, Union

import aiohttp

from app.core.llm.base import ChatChunk, ChatModel, ChatResponse, EmbeddingProvider, EmbeddingResponse

logger = logging.getLogger(__name__)


class OpenAICompatChatModel(ChatModel):
    """Chat completion via OpenAI-compatible /chat/completions endpoint."""

    def __init__(self, model_id: str, base_url: str, api_key: str, timeout: int = 300):
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = aiohttp.ClientTimeout(total=timeout)

    # ── private helpers ────────────────────────────────────────

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _chat_url(self) -> str:
        # Most providers: /chat/completions
        # Tongyi also uses this path at compatible-mode base URL
        return f"{self.base_url}/chat/completions"

    def _build_body(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        stream: bool,
    ) -> dict:
        return {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

    async def _parse_non_stream(self, data: dict) -> ChatResponse:
        """Parse a non-streaming response JSON into ChatResponse."""
        choice = data["choices"][0]
        return ChatResponse(
            content=choice["message"]["content"],
            model=data.get("model", self.model_id),
            usage=data.get("usage", {}),
        )

    async def _parse_stream_line(self, line: str) -> ChatChunk | None:
        """Parse a single SSE data line into ChatChunk (or None if [DONE])."""
        if not line.startswith("data: "):
            return None
        payload = line[len("data: "):]
        if payload.strip() == "[DONE]":
            return None
        try:
            data = json.loads(payload)
            choice = data["choices"][0]
            delta = choice.get("delta", {})
            content = delta.get("content", "") or ""
            return ChatChunk(
                delta_content=content,
                finish_reason=choice.get("finish_reason"),
            )
        except (json.JSONDecodeError, KeyError, IndexError):
            return None

    # ── public API ─────────────────────────────────────────────

    async def achat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False,
    ) -> Union[ChatResponse, AsyncIterator[ChatChunk]]:
        body = self._build_body(messages, temperature, max_tokens, stream)

        if stream:
            return self._achat_stream(body)

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.post(self._chat_url(), headers=self._headers(), json=body) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"LLM API error {resp.status}: {text[:500]}")
                data = await resp.json()
                return await self._parse_non_stream(data)

    async def _achat_stream(self, body: dict) -> AsyncIterator[ChatChunk]:
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.post(self._chat_url(), headers=self._headers(), json=body) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"LLM API error {resp.status}: {text[:500]}")
                async for line in resp.content:
                    text_line = line.decode("utf-8").strip()
                    if not text_line:
                        continue
                    chunk = await self._parse_stream_line(text_line)
                    if chunk is not None:
                        yield chunk


class OpenAICompatEmbedding(EmbeddingProvider):
    """Embedding via OpenAI-compatible /embeddings endpoint."""

    def __init__(self, model_id: str, base_url: str, api_key: str):
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _url(self) -> str:
        return f"{self.base_url}/embeddings"

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def aembed(self, texts: list[str]) -> EmbeddingResponse:
        # Tongyi embedding API uses "input" as the key
        body = {
            "model": self.model_id,
            "input": texts,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(self._url(), headers=self._headers(), json=body) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"Embedding API error {resp.status}: {text[:500]}")
                data = await resp.json()

                # Normalize: Tongyi uses data[].embedding, OpenAI-compat same
                embeddings = [item["embedding"] for item in data["data"]]
                # Sort by index if present
                if data["data"] and "index" in data["data"][0]:
                    embeddings = [
                        item["embedding"]
                        for item in sorted(data["data"], key=lambda x: x["index"])
                    ]
                return EmbeddingResponse(
                    embeddings=embeddings,
                    model=data.get("model", self.model_id),
                    usage=data.get("usage", {}),
                )
