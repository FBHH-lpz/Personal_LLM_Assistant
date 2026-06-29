import gradio as gr

def dummy_chat(message, history):
    print(f"[DEBUG] message type: {type(message)}, value: {message!r}")
    print(f"[DEBUG] history type: {type(history)}, value: {history!r}")
    return "Dummy response"

demo = gr.ChatInterface(fn=dummy_chat)
demo.launch(server_port=7861, prevent_thread_lock=True)

import time
import requests

time.sleep(2)
try:
    from gradio_client import Client
    client = Client("http://127.0.0.1:7861/")
    res = client.predict(message="hello", api_name="/chat")
    print("API response 1:", res)
    res = client.predict(message="how are you", api_name="/chat")
    print("API response 2:", res)
except Exception as e:
    import traceback
    traceback.print_exc()

import sys
sys.exit(0)
