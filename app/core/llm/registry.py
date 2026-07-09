"""Provider registry — factory functions for creating LLM/embedding clients."""

from __future__ import annotations

import logging
from functools import cache

from app.config import Settings
from app.core.llm.base import ChatModel, EmbeddingProvider
from app.core.llm.deepseek import create_deepseek_chat
from app.core.llm.tongyi import create_tongyi_chat, create_tongyi_embedding
from app.core.llm.zhipu import create_zhipu_chat

logger = logging.getLogger(__name__)


@cache
def get_chat_model(settings: Settings | None = None) -> ChatModel:
    """Return the configured chat model (cached per process)."""
    if settings is None:
        from app.config import settings as _s
        settings = _s

    provider = settings.llm_provider
    model_id = settings.llm_model

    logger.info("Creating chat model: provider=%s model=%s", provider, model_id)

    if provider == "tongyi":
        return create_tongyi_chat(
            model_id=model_id,
            api_key=settings.tongyi_api_key,
            base_url=settings.tongyi_base_url,
        )
    elif provider == "deepseek":
        return create_deepseek_chat(
            model_id=model_id,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )
    elif provider == "zhipu":
        return create_zhipu_chat(
            model_id=model_id,
            api_key=settings.zhipu_api_key,
            base_url=settings.zhipu_base_url,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


@cache
def get_cheap_chat_model(settings: Settings | None = None) -> ChatModel:
    """Return the cheap model for lightweight tasks like query rewriting."""
    if settings is None:
        from app.config import settings as _s
        settings = _s

    # DeepSeek's chat model is already cheap; for others, use turbo models
    provider = settings.llm_provider
    cheap_model = settings.llm_model_cheap

    if provider == "tongyi":
        return create_tongyi_chat(
            model_id=cheap_model or "qwen-turbo",
            api_key=settings.tongyi_api_key,
            base_url=settings.tongyi_base_url,
        )
    elif provider == "deepseek":
        return create_deepseek_chat(
            model_id=cheap_model or "deepseek-chat",
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )
    elif provider == "zhipu":
        return create_zhipu_chat(
            model_id=cheap_model or "glm-4-flash",
            api_key=settings.zhipu_api_key,
            base_url=settings.zhipu_base_url,
        )
    else:
        return get_chat_model(settings)


@cache
def get_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    """Return the configured embedding provider (cached per process)."""
    if settings is None:
        from app.config import settings as _s
        settings = _s

    # Currently only Tongyi provides embeddings
    provider = settings.embedding_provider

    if provider == "tongyi":
        return create_tongyi_embedding(
            model_id=settings.tongyi_embedding_model,
            api_key=settings.tongyi_api_key,
            base_url=settings.tongyi_base_url,
        )
    else:
        raise ValueError(f"Embedding provider not supported: {provider}")
