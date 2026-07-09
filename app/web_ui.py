"""Gradio chat UI for the RAG assistant."""

import json

import gradio as gr
import httpx

API_BASE = "http://localhost:8000"


async def chat_with_assistant(message: str, history: list, conv_id_state: str):
    """Send message to RAG API and yield streaming response."""
    conv_id = conv_id_state

    async with httpx.AsyncClient(timeout=300) as client:
        body = {"content": message, "user_id": "default"}
        if conv_id:
            body["conversation_id"] = conv_id

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

                if "conversation_id" in data and not conv_id:
                    conv_id = data["conversation_id"]
                if "delta" in data:
                    full_response += data["delta"]
                    yield full_response, conv_id

        if not full_response:
            yield "（无响应，请确认 API 服务已启动）", conv_id


# ── UI ────────────────────────────────────────────────────────

with gr.Blocks(title="Personal LLM Assistant") as demo:
    gr.Markdown("# Personal LLM Assistant")
    gr.Markdown("RAG 知识库问答 — 混合检索 + Query Rewriting + 流式输出")

    conv_id = gr.State("")

    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(placeholder="输入问题...", show_label=False)

    async def respond(message, chat_history, cid):
        chat_history = chat_history or []
        chat_history.append({"role": "user", "content": message})

        full_text = ""
        new_cid = cid
        async for text, ncid in chat_with_assistant(message, chat_history, cid):
            full_text = text
            new_cid = ncid

        chat_history.append({"role": "assistant", "content": full_text})
        return "", chat_history, new_cid

    msg.submit(respond, [msg, chatbot, conv_id], [msg, chatbot, conv_id])

    gr.Markdown("API: [http://localhost:8000/docs](http://localhost:8000/docs)")


if __name__ == "__main__":
    demo.queue().launch(server_port=7860)
