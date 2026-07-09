"""Gradio chat UI for the RAG assistant."""

import json

import gradio as gr
import httpx

API_BASE = "http://localhost:8000"


async def stream_chat(message: str, history: list):
    """Generator that yields streamed responses for Gradio ChatInterface."""
    conv_id = ""
    # Extract conversation_id from previous turns (stored in extra state)
    # For now, each message starts fresh

    async with httpx.AsyncClient(timeout=300) as client:
        body = {"content": message, "user_id": "default"}
        # Use conversation_id from session if we had it
        # For simplicity, let API auto-create

        full_response = ""

        async with client.stream("POST", f"{API_BASE}/chat", json=body) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                if "delta" in data:
                    full_response += data["delta"]
                    yield full_response

        if not full_response:
            yield "（无响应）"


# ── UI ────────────────────────────────────────────────────────

demo = gr.ChatInterface(
    fn=stream_chat,
    title="Personal LLM Assistant",
    description="RAG 知识库问答 — 混合检索 + Multi-Query + 流式输出",
    chatbot=gr.Chatbot(height=500),
)

if __name__ == "__main__":
    demo.queue().launch(server_port=7860)
