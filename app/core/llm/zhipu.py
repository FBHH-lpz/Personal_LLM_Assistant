"""Zhipu (GLM) provider."""

from app.core.llm._openai_compat import OpenAICompatChatModel
from app.core.llm.base import ChatModel


def create_zhipu_chat(
    model_id: str = "glm-4-flash",
    api_key: str = "",
    base_url: str = "https://open.bigmodel.cn/api/paas/v4",
) -> ChatModel:
    return OpenAICompatChatModel(model_id=model_id, base_url=base_url, api_key=api_key)
