from transformers import AutoTokenizer

try:
    tokenizer = AutoTokenizer.from_pretrained("qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    tools = [{
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                },
                "required": ["location"],
            },
        }
    }]
    messages = [{"role": "user", "content": "What's the weather like in Boston?"}]
    text = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
    print("TOOL CALLING TEMPLATE:")
    print(text)
except Exception as e:
    print(f"Error: {e}")
