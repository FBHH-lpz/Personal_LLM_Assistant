"""Tongyi (Qwen) provider via DashScope compatible-mode API."""

from app.core.llm._openai_compat import OpenAICompatChatModel, OpenAICompatEmbedding
from app.core.llm.base import ChatModel, EmbeddingProvider


def create_tongyi_chat(
    model_id: str = "qwen-turbo",
    api_key: str = "",
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
) -> ChatModel:
    return OpenAICompatChatModel(model_id=model_id, base_url=base_url, api_key=api_key)


def create_tongyi_embedding(
    model_id: str = "text-embedding-v3",
    api_key: str = "",
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
) -> EmbeddingProvider:
    return OpenAICompatEmbedding(model_id=model_id, base_url=base_url, api_key=api_key)
