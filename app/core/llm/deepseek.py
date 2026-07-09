"""DeepSeek provider."""

from app.core.llm._openai_compat import OpenAICompatChatModel
from app.core.llm.base import ChatModel


def create_deepseek_chat(
    model_id: str = "deepseek-chat",
    api_key: str = "",
    base_url: str = "https://api.deepseek.com/v1",
) -> ChatModel:
    return OpenAICompatChatModel(model_id=model_id, base_url=base_url, api_key=api_key)
